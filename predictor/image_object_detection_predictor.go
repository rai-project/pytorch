package predictor

import (
	"bufio"
	"context"
	"io"
	"os"
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

// ObjectDetectionPredictor ...
type ObjectDetectionPredictor struct {
	common.ImagePredictor
	predictor          *gopytorch.Predictor
	labels             []string
	inputLayer         string
	boxesLayer         string
	probabilitiesLayer string
	classesLayer       string
	boxes              interface{}
	probabilities      interface{}
	classes            interface{}
}

// New ...
func NewObjectDetectionPredictor(model dlframework.ModelManifest, opts ...options.Option) (common.Predictor, error) {
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

	predictor := new(ObjectDetectionPredictor)

	return predictor.Load(ctx, model, opts...)
}

// Download ...
func (p *ObjectDetectionPredictor) Download(ctx context.Context, model dlframework.ModelManifest, opts ...options.Option) error {
	framework, err := model.ResolveFramework()
	if err != nil {
		return err
	}

	workDir, err := model.WorkDir()
	if err != nil {
		return err
	}

	ip := &ObjectDetectionPredictor{
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
func (p *ObjectDetectionPredictor) Load(ctx context.Context, model dlframework.ModelManifest, opts ...options.Option) (common.Predictor, error) {
	framework, err := model.ResolveFramework()
	if err != nil {
		return nil, err
	}

	workDir, err := model.WorkDir()
	if err != nil {
		return nil, err
	}

	ip := &ObjectDetectionPredictor{
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

func (p *ObjectDetectionPredictor) download(ctx context.Context) error {
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

	span.LogFields(
		olog.String("event", "download features"),
	)
	checksum := p.GetFeaturesChecksum()
	if checksum != "" {
		if _, err := downloadmanager.DownloadFile(p.GetFeaturesUrl(), p.GetFeaturesPath(), downloadmanager.MD5Sum(checksum)); err != nil {
			return err
		}
	} else {
		if _, err := downloadmanager.DownloadFile(p.GetFeaturesUrl(), p.GetFeaturesPath()); err != nil {
			return err
		}
	}

	return nil
}

func (p *ObjectDetectionPredictor) loadPredictor(ctx context.Context) error {
	span, ctx := tracer.StartSpanFromContext(ctx, tracer.APPLICATION_TRACE, "load_predictor")
	defer span.Finish()

	span.LogFields(
		olog.String("event", "read features"),
	)

	var labels []string
	f, err := os.Open(p.GetFeaturesPath())
	if err != nil {
		return errors.Wrapf(err, "cannot read %s", p.GetFeaturesPath())
	}
	defer f.Close()
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := scanner.Text()
		labels = append(labels, line)
	}
	p.labels = labels

	span.LogFields(
		olog.String("event", "creating predictor"),
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

func (p *ObjectDetectionPredictor) GetInputLayerName(reader io.Reader, layer string) (string, error) {
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

func (p *ObjectDetectionPredictor) GetOutputLayerName(reader io.Reader, layer string) (string, error) {
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

// Predict ...
func (p *ObjectDetectionPredictor) Predict(ctx context.Context, data interface{}, opts ...options.Option) error {
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

	// TODO support data types other than float32
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
func (p *ObjectDetectionPredictor) ReadPredictedFeatures(ctx context.Context) ([]dlframework.Features, error) {
	span, ctx := tracer.StartSpanFromContext(ctx, tracer.APPLICATION_TRACE, "read_predicted_features")
	defer span.Finish()

	// TODO
	outputs, err := p.predictor.ReadPredictionOutput(ctx)
	if err != nil {
		return nil, err
	}
	// would be getting two tensors as outputs since
	// we are performing object detection
	// scores = outputs[0], dimensions of boxes = outputs[1]
	//boxes := outputs[1].Data().([]float32)
	//probabilities := outputs[0].Data().([]float32)
	//classes := p.classes.([][]float32)

	// Dummy declarations
	//boxes := p.boxes.([][][]float32)
	//probabilities := p.probabilities.([][]float32)
	classes := p.classes.([][]float32)
	return p.CreateBoundingBoxFeatures(ctx, outputs[0], classes, outputs[1], p.labels)
}

// Reset ...
func (p *ObjectDetectionPredictor) Reset(ctx context.Context) error {
	return nil
}

// Close ...
func (p *ObjectDetectionPredictor) Close() error {
	if p.predictor != nil {
		p.predictor.Close()
	}
	return nil
}

func (p *ObjectDetectionPredictor) Modality() (dlframework.Modality, error) {
	return dlframework.ImageObjectDetectionModality, nil
}

func init() {
	config.AfterInit(func() {
		framework := pytorch.FrameworkManifest
		agent.AddPredictor(framework, &ObjectDetectionPredictor{
			ImagePredictor: common.ImagePredictor{
				Base: common.Base{
					Framework: framework,
				},
			},
		})
	})
}
