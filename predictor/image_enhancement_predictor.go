package predictor

// import (
// 	"bufio"
// 	"context"
// 	"io"
// 	"os"
// 	"strings"

// 	opentracing "github.com/opentracing/opentracing-go"
// 	olog "github.com/opentracing/opentracing-go/log"
// 	"github.com/pkg/errors"
// 	"github.com/rai-project/config"
// 	"github.com/rai-project/dlframework"
// 	"github.com/rai-project/dlframework/framework/agent"
// 	"github.com/rai-project/dlframework/framework/options"
// 	common "github.com/rai-project/dlframework/framework/predictor"
// 	"github.com/rai-project/downloadmanager"
// 	gopytorch "github.com/rai-project/go-pytorch"
// 	"github.com/rai-project/pytorch"
// 	"github.com/rai-project/tracer"
// 	gotensor "gorgonia.org/tensor"
// )

// type ImageEnhancementPredictor struct {
// 	common.ImagePredictor
// 	predictor *gopytorch.Predictor
// 	images    interface{}
// }

// func NewImageEnhancementPredictor(model dlframework.ModelManifest, opts ...options.Option) (common.Predictor, error) {
// 	ctx := context.Background()
// 	span, ctx := tracer.StartSpanFromContext(ctx, tracer.APPLICATION_TRACE, "new_predictor")
// 	defer span.Finish()

// 	modelInputs := model.GetInputs()
// 	if len(modelInputs) != 1 {
// 		return nil, errors.New("number of inputs not supported")
// 	}
// 	firstInputType := modelInputs[0].GetType()
// 	if strings.ToLower(firstInputType) != "image" {
// 		return nil, errors.New("input type not supported")
// 	}

// 	predictor := new(ImageEnhancementPredictor)

// 	return predictor.Load(context.Background(), model, opts...)
// }

// // Download ...
// func (p *ImageEnhancementPredictor) Download(ctx context.Context, model dlframework.ModelManifest, opts ...options.Option) error {
// 	framework, err := model.ResolveFramework()
// 	if err != nil {
// 		return err
// 	}

// 	workDir, err := model.WorkDir()
// 	if err != nil {
// 		return err
// 	}

// 	ip := &ImageEnhancementPredictor{
// 		ImagePredictor: common.ImagePredictor{
// 			Base: common.Base{
// 				Framework: framework,
// 				Model:     model,
// 				WorkDir:   workDir,
// 				Options:   options.New(opts...),
// 			},
// 		},
// 	}

// 	if err = p.download(ctx); err != nil {
// 		return err
// 	}

// 	return nil
// }

// func (p *ImageEnhancementPredictor) Load(ctx context.Context, model dlframework.ModelManifest, opts ...options.Option) (common.Predictor, error) {
// 	framework, err := model.ResolveFramework()
// 	if err != nil {
// 		return nil, err
// 	}

// 	workDir, err := model.WorkDir()
// 	if err != nil {
// 		return nil, err
// 	}

// 	ip := &ImageEnhancementPredictor{
// 		ImagePredictor: common.ImagePredictor{
// 			Base: common.Base{
// 				Framework: framework,
// 				Model:     model,
// 				WorkDir:   workDir,
// 				Options:   options.New(opts...),
// 			},
// 		},
// 	}

// 	if err = ip.download(ctx); err != nil {
// 		return nil, err
// 	}

// 	if err = ip.loadPredictor(ctx); err != nil {
// 		return nil, err
// 	}

// 	return ip, nil
// }

// func (p *ImageEnhancementPredictor) download(ctx context.Context) error {
// 	span, ctx := opentracing.StartSpanFromContext(
// 		ctx,
// 		"download",
// 		opentracing.Tags{
// 			"graph_url":           p.GetGraphUrl(),
// 			"target_graph_file":   p.GetGraphPath(),
// 			"weights_url":         p.GetWeightsUrl(),
// 			"target_weights_file": p.GetWeightsPath(),
// 		},
// 	)
// 	defer span.Finish()

// 	model := p.Model
// 	if model.Model.IsArchive {
// 		baseURL := model.Model.BaseUrl
// 		span.LogFields(
// 			olog.String("event", "download model archive"),
// 		)
// 		_, err := downloadmanager.DownloadInto(baseURL, p.WorkDir, downloadmanager.Context(ctx))
// 		if err != nil {
// 			return errors.Wrapf(err, "failed to download model archive from %v", model.Model.BaseUrl)
// 		}
// 	} else {
// 		span.LogFields(
// 			olog.String("event", "download graph"),
// 		)
// 		checksum := p.GetGraphChecksum()
// 		if checksum != "" {
// 			if _, err := downloadmanager.DownloadFile(p.GetGraphUrl(), p.GetGraphPath(), downloadmanager.MD5Sum(checksum)); err != nil {
// 				return err
// 			}
// 		} else {
// 			if _, err := downloadmanager.DownloadFile(p.GetGraphUrl(), p.GetGraphPath()); err != nil {
// 				return err
// 			}
// 		}
// 	}

// 	return nil
// }

// func (p *ImageEnhancementPredictor) loadPredictor(ctx context.Context) error {
// 	span, ctx := tracer.StartSpanFromContext(ctx, tracer.APPLICATION_TRACE, "load_predictor")
// 	defer span.Finish()

// 	span.LogFields(
// 		olog.String("event", "load predictor"),
// 	)

// 	opts, err := p.GetPredictionOptions(ctx)
// 	if err != nil {
// 		return err
// 	}

// 	pred, err := gopytorch.New(
// 		ctx,
// 		options.WithOptions(opts),
// 		options.Graph([]byte(p.GetGraphPath())),
// 	)
// 	if err != nil {
// 		return err
// 	}

// 	p.predictor = pred

// 	return nil
// }

// func (p ImageEnhancementPredictor) GetInputLayerName(reader io.Reader, layer string) (string, error) {
// 	model := p.Model
// 	modelInputs := model.GetInputs()
// 	typeParameters := modelInputs[0].GetParameters()
// 	name, err := p.GetTypeParameter(typeParameters, layer)
// 	if err != nil {
// 		// TODO get input layer name => for what..?
// 		return "", errors.New("cannot determine the name of the input layer")
// 	}
// 	return name, nil
// }

// func (p ImageEnhancementPredictor) GetOutputLayerName(reader io.Reader, layer string) (string, error) {
// 	model := p.Model
// 	modelOutput := model.GetOutput()
// 	typeParameters := modelOutput.GetParameters()
// 	name, err := p.GetTypeParameter(typeParameters, layer)
// 	if err != nil {
// 		// TODO get output layer name => for what..?
// 		return "", errors.New("cannot determine the name of the output layer")
// 	}
// 	return name, nil
// }

// func (p *ImageEnhancementPredictor) runOptions() *proto.RunOptions {
// 	if p.TraceLevel() >= tracer.FRAMEWORK_TRACE {
// 		return &proto.RunOptions{
// 			TraceLevel: proto.RunOptions_SOFTWARE_TRACE,
// 		}
// 	}
// 	return nil
// }

// func makeUniformImage() [][][][]float32 {
// 	images := make([][][][]float32, 10)
// 	width := 1000
// 	height := 1000
// 	for ii := range images {
// 		sl := make([][][]float32, height)
// 		for jj := range sl {
// 			el := make([][]float32, width)
// 			for kk := range el {
// 				el[kk] = []float32{1, 0, 1}
// 			}
// 			sl[jj] = el
// 		}
// 		images[ii] = sl
// 	}
// 	return images
// }

// // Predict ...
// func (p *ImageEnhancementPredictor) Predict(ctx context.Context, data interface{}, opts ...options.Option) error {
// 	// p.images = makeUniformImage()

// 	// return nil

// 	span, ctx := tracer.StartSpanFromContext(ctx, tracer.APPLICATION_TRACE, "predict")
// 	defer span.Finish()

// 	if data == nil {
// 		return errors.New("input data nil")
// 	}
// 	input, ok := data.([]*gotensor.Dense)
// 	if !ok {
// 		return errors.New("input data is not slice of dense tensors")
// 	}

// 	session := p.tfSession
// 	graph := p.tfGraph

// 	tensor, err := makeTensorFromGoTensors(input)
// 	if err != nil {
// 		return err
// 	}

// 	sessionSpan, ctx := tracer.StartSpanFromContext(ctx, tracer.APPLICATION_TRACE, "c_predict")
// 	// TODO interface image enhancement call to predictor backend
// 	return nil
// }

// // ReadPredictedFeatures ...
// func (p *ImageEnhancementPredictor) ReadPredictedFeatures(ctx context.Context) ([]dlframework.Features, error) {
// 	span, ctx := tracer.StartSpanFromContext(ctx, tracer.APPLICATION_TRACE, "read_predicted_features")
// 	defer span.Finish()

// 	e, ok := p.images.([][][][]float32)
// 	if !ok {
// 		return nil, errors.New("output is not of type [][][][]float32")
// 	}
// 	return p.CreateRawImageFeatures(ctx, e)
// }

// func (p *ImageEnhancementPredictor) Reset(ctx context.Context) error {

// 	return nil
// }

// func (p *ImageEnhancementPredictor) Close() error {
// 	return nil
// }

// func (p ImageEnhancementPredictor) Modality() (dlframework.Modality, error) {
// 	return dlframework.ImageEnhancementModality, nil
// }

// func init() {
// 	config.AfterInit(func() {
// 		framework := pytorch.FrameworkManifest
// 		agent.AddPredictor(framework, &ImageEnhancementPredictor{
// 			ImagePredictor: common.ImagePredictor{
// 				Base: common.Base{
// 					Framework: framework,
// 				},
// 			},
// 		})
// 	})
// }
