package predictor

import (
	"context"
	"io"
	"strings"

	"github.com/k0kubun/pp"
	opentracing "github.com/opentracing/opentracing-go"
	olog "github.com/opentracing/opentracing-go/log"
	"github.com/pkg/errors"
	"github.com/rai-project/config"
	"github.com/rai-project/dlframework"
	"github.com/rai-project/dlframework/framework/agent"
	"github.com/rai-project/dlframework/framework/options"
	common "github.com/rai-project/dlframework/framework/predictor"
	"github.com/rai-project/downloadmanager"
	gopytorch "github.com/rai-project/go-pytorch"
	"github.com/rai-project/pytorch"
	"github.com/rai-project/tracer"
	"github.com/rai-project/tracer/ctimer"
	gotensor "gorgonia.org/tensor"
)

// ImageEnhancementPredictor ...
type ImageEnhancementPredictor struct {
	common.ImagePredictor
	predictor *gopytorch.Predictor
	images    interface{}
}

// New ...
func NewImageEnhancementPredictor(model dlframework.ModelManifest, opts ...options.Option) (common.Predictor, error) {
	ctx := context.Background()
	span, ctx := tracer.StartSpanFromContext(ctx, tracer.APPLICATION_TRACE, "new_predictor")
	defer span.Finish()

	modelInputs := model.GetInputs()
	if len(modelInputs) != 1 {
		return nil, errors.New("number of inputs not supported")
	}
	firstInputType := modelInputs[0].GetType()
	if strings.ToLower(firstInputType) != "image" {
		return nil, errors.New("input type not supported")
	}

	predictor := new(ImageEnhancementPredictor)

	return predictor.Load(context.Background(), model, opts...)
}

// Download ...
func (p *ImageEnhancementPredictor) Download(ctx context.Context, model dlframework.ModelManifest, opts ...options.Option) error {
	framework, err := model.ResolveFramework()
	if err != nil {
		return err
	}

	workDir, err := model.WorkDir()
	if err != nil {
		return err
	}

	ip := &ImageEnhancementPredictor{
		ImagePredictor: common.ImagePredictor{
			Base: common.Base{
				Framework: framework,
				Model:     model,
				WorkDir:   workDir,
				Options:   options.New(opts...),
			},
		},
	}

	if err = ip.download(ctx); err != nil {
		return err
	}

	return nil
}

// Load ...
func (p *ImageEnhancementPredictor) Load(ctx context.Context, model dlframework.ModelManifest, opts ...options.Option) (common.Predictor, error) {
	framework, err := model.ResolveFramework()
	if err != nil {
		return nil, err
	}

	workDir, err := model.WorkDir()
	if err != nil {
		return nil, err
	}

	ip := &ImageEnhancementPredictor{
		ImagePredictor: common.ImagePredictor{
			Base: common.Base{
				Framework: framework,
				Model:     model,
				WorkDir:   workDir,
				Options:   options.New(opts...),
			},
		},
	}

	if err = ip.download(ctx); err != nil {
		return nil, err
	}

	if err = ip.loadPredictor(ctx); err != nil {
		return nil, err
	}

	return ip, nil
}

func (p *ImageEnhancementPredictor) download(ctx context.Context) error {
	span, ctx := tracer.StartSpanFromContext(
		ctx,
		tracer.APPLICATION_TRACE,
		"download",
		opentracing.Tags{
			"graph_url":           p.GetGraphUrl(),
			"target_graph_file":   p.GetGraphPath(),
			"feature_url":         p.GetFeaturesUrl(),
			"target_feature_file": p.GetFeaturesPath(),
		},
	)
	defer span.Finish()

	model := p.Model
	if model.Model.IsArchive {
		baseURL := model.Model.BaseUrl
		span.LogFields(
			olog.String("event", "download model archive"),
		)
		_, err := downloadmanager.DownloadInto(baseURL, p.WorkDir, downloadmanager.Context(ctx))
		if err != nil {
			return errors.Wrapf(err, "failed to download model archive from %v", model.Model.BaseUrl)
		}
	} else {
		span.LogFields(
			olog.String("event", "download graph"),
		)
		checksum := p.GetGraphChecksum()
		if checksum != "" {
			if _, err := downloadmanager.DownloadFile(p.GetGraphUrl(), p.GetGraphPath(), downloadmanager.MD5Sum(checksum)); err != nil {
				return err
			}
		} else {
			if _, err := downloadmanager.DownloadFile(p.GetGraphUrl(), p.GetGraphPath()); err != nil {
				return err
			}
		}
	}

	return nil
}

func (p *ImageEnhancementPredictor) loadPredictor(ctx context.Context) error {
	span, ctx := tracer.StartSpanFromContext(ctx, tracer.APPLICATION_TRACE, "load_predictor")
	defer span.Finish()

	span.LogFields(
		olog.String("event", "load predictor"),
	)

	opts, err := p.GetPredictionOptions()
	if err != nil {
		return err
	}

	pred, err := gopytorch.New(
		ctx,
		options.WithOptions(opts),
		options.Graph([]byte(p.GetGraphPath())),
	)
	if err != nil {
		return err
	}

	p.predictor = pred

	return nil
}

func (p ImageEnhancementPredictor) GetInputLayerName(reader io.Reader, layer string) (string, error) {
	model := p.Model
	modelInputs := model.GetInputs()
	typeParameters := modelInputs[0].GetParameters()
	name, err := p.GetTypeParameter(typeParameters, layer)
	if err != nil {
		// TODO get input layer name => for what..?
		return "", errors.New("cannot determine the name of the input layer")
	}
	return name, nil
}

func (p ImageEnhancementPredictor) GetOutputLayerName(reader io.Reader, layer string) (string, error) {
	model := p.Model
	modelOutput := model.GetOutput()
	typeParameters := modelOutput.GetParameters()
	name, err := p.GetTypeParameter(typeParameters, layer)
	if err != nil {
		// TODO get output layer name => for what..?
		return "", errors.New("cannot determine the name of the output layer")
	}
	return name, nil
}

func makeUniformImage() [][][][]float32 {
	images := make([][][][]float32, 10)
	width := 1000
	height := 1000
	for ii := range images {
		sl := make([][][]float32, height)
		for jj := range sl {
			el := make([][]float32, width)
			for kk := range el {
				el[kk] = []float32{1, 0, 1}
			}
			sl[jj] = el
		}
		images[ii] = sl
	}
	return images
}

// Predict ...
func (p *ImageEnhancementPredictor) Predict(ctx context.Context, data interface{}, opts ...options.Option) error {

	if p.TraceLevel() >= tracer.FRAMEWORK_TRACE {
		p.predictor.EnableProfiling()
		err := p.predictor.StartProfiling("pytorch", "predict")
		if err != nil {
			log.WithError(err).WithField("framework", "pytorch").Error("unable to start framework profiling")
		} else {
			defer func() {
				p.predictor.EndProfiling()
				profBuffer, err := p.predictor.ReadProfile()
				if err != nil {
					pp.Println(err)
					return
				}
				t, err := ctimer.New(profBuffer)
				if err != nil {
					panic(err)
					return
				}
				t.Publish(ctx, tracer.FRAMEWORK_TRACE)
				p.predictor.DisableProfiling()
			}()
		}
	}

	if data == nil {
		return errors.New("input data nil")
	}

	gotensors, ok := data.([]*gotensor.Dense)
	if !ok {
		return errors.New("input data is not slice of dense tensors")
	}

	fst := gotensors[0]
	dims := append([]int{len(gotensors)}, fst.Shape()...)
	// debug
	pp.Println(dims)
	// TODO: support data types other than float32
	var input []float32
	for _, t := range gotensors {
		input = append(input, t.Float32s()...)
	}

	err := p.predictor.Predict(ctx, []gotensor.Tensor{
		gotensor.New(
			gotensor.Of(gotensor.Float32),
			gotensor.WithBacking(input),
			gotensor.WithShape(dims...),
		),
	})
	if err != nil {
		return err
	}

	return nil

}

// ReadPredictedFeatures ...
func (p *ImageEnhancementPredictor) ReadPredictedFeatures(ctx context.Context) ([]dlframework.Features, error) {
	span, ctx := tracer.StartSpanFromContext(ctx, tracer.APPLICATION_TRACE, "read_predicted_features")
	defer span.Finish()

	outputs, err := p.predictor.ReadPredictionOutput(ctx)
	if err != nil {
		return nil, err
	}
	// would be getting only one tensor as outputs since
	// we are performing image classification
	//output := outputs[0].Data().([]float32)
	//if err != nil {
	//  return nil, err
	//}
	output_array := outputs[0].Data().([]float32)
	output_batch := outputs[0].Shape()[0]
	output_channels := outputs[0].Shape()[1]
	output_height := outputs[0].Shape()[2]
	output_width := outputs[1].Shape()[3]
	// convert 1D array to a 4D array in order to make it compatible with CreateRawImageFeatures function call
	e := make([][][][]float32, output_batch)
	for b := 0; b < output_batch; b++ {
		e[b] = make([][][]float32, output_height)
		for h := 0; h < output_height; h++ {
			e[b][h] = make([][]float32, output_width)
			for w := 0; w < output_width; w++ {
				e[b][h][w] = make([]float32, 3)
			}
		}
	}
	for b := 0; b < output_batch; b++ {
		for h := 0; h < output_height; h++ {
			for w := 0; w < output_width; w++ {
				e[b][h][w][0] = output_array[b*(output_height*output_width*output_channels)+h*output_width+w*output_channels+0]
				e[b][h][w][1] = output_array[b*(output_height*output_width*output_channels)+h*output_width+w*output_channels+1]
				e[b][h][w][2] = output_array[b*(output_height*output_width*output_channels)+h*output_width+w*output_channels+2]
			}
		}
	}

	return p.CreateRawImageFeatures(ctx, e)
}

func (p *ImageEnhancementPredictor) Reset(ctx context.Context) error {

	return nil
}

func (p *ImageEnhancementPredictor) Close() error {
	return nil
}

func (p ImageEnhancementPredictor) Modality() (dlframework.Modality, error) {
	return dlframework.ImageEnhancementModality, nil
}

func init() {
	config.AfterInit(func() {
		framework := pytorch.FrameworkManifest
		agent.AddPredictor(framework, &ImageEnhancementPredictor{
			ImagePredictor: common.ImagePredictor{
				Base: common.Base{
					Framework: framework,
				},
			},
		})
	})
}
