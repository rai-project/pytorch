package predictor

import (
	"bufio"
	"bytes"
	"context"
	"io"
	"io/ioutil"
	"os"
	"runtime"
	"strings"

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
	nvidiasmi "github.com/rai-project/nvidia-smi"
	"github.com/rai-project/pytorch"
	"github.com/rai-project/tracer"
	gotensor "gorgonia.org/tensor"
)

type InstanceSegmentationPredictor struct {
	common.ImagePredictor
	labels             []string
	inputLayer         string
	boxesLayer         string
	probabilitiesLayer string
	classesLayer       string
	masksLayer         string
	boxes              interface{}
	probabilities      interface{}
	classes            interface{}
	masks              interface{}
}

func NewInstanceSegmentationPredictor(model dlframework.ModelManifest, opts ...options.Option) (common.Predictor, error) {
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

	predictor := new(InstanceSegmentationPredictor)

	return predictor.Load(context.Background(), model, opts...)
}

// Download ...
func (p *InstanceSegmentationPredictor) Download(ctx context.Context, model dlframework.ModelManifest, opts ...options.Option) error {
	framework, err := model.ResolveFramework()
	if err != nil {
		return err
	}

	workDir, err := model.WorkDir()
	if err != nil {
		return err
	}

	ip := &InstanceSegmentationPredictor{
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

func (p *InstanceSegmentationPredictor) Load(ctx context.Context, model dlframework.ModelManifest, opts ...options.Option) (common.Predictor, error) {
	framework, err := model.ResolveFramework()
	if err != nil {
		return nil, err
	}

	workDir, err := model.WorkDir()
	if err != nil {
		return nil, err
	}

	ip := &InstanceSegmentationPredictor{
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

func (p *InstanceSegmentationPredictor) download(ctx context.Context) error {
	span, ctx := opentracing.StartSpanFromContext(
		ctx,
		"download",
		opentracing.Tags{
			"graph_url":           p.GetGraphUrl(),
			"target_graph_file":   p.GetGraphPath(),
			"weights_url":         p.GetWeightsUrl(),
			"target_weights_file": p.GetWeightsPath(),
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

func (p *InstanceSegmentationPredictor) loadPredictor(ctx context.Context) error {
	span, ctx := tracer.StartSpanFromContext(ctx, tracer.APPLICATION_TRACE, "load_predictor")
	defer span.Finish()

	if p.tfSession != nil {
		return nil
	}

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
		olog.String("event", "read graph"),
	)
	model, err := ioutil.ReadFile(p.GetGraphPath())
	if err != nil {
		return errors.Wrapf(err, "cannot read %s", p.GetGraphPath())
	}

	modelReader := bytes.NewReader(model)
	p.inputLayer, err = p.GetInputLayerName(modelReader, "input_layer")
	if err != nil {
		return errors.Wrap(err, "failed to get input layer name")
	}
	p.boxesLayer, err = p.GetOutputLayerName(modelReader, "boxes_layer")
	if err != nil {
		return errors.Wrap(err, "failed to get the boxes layer name")
	}
	p.probabilitiesLayer, err = p.GetOutputLayerName(modelReader, "probabilities_layer")
	if err != nil {
		return errors.Wrap(err, "failed to get the probabilities layer name")
	}
	p.classesLayer, err = p.GetOutputLayerName(modelReader, "classes_layer")
	if err != nil {
		return errors.Wrap(err, "failed to get the classes layer name")
	}
	p.masksLayer, err = p.GetOutputLayerName(modelReader, "masks_layer")
	if err != nil {
		return errors.Wrap(err, "failed to get the masks layer name")
	}

	return nil
}

func (p InstanceSegmentationPredictor) GetInputLayerName(reader io.Reader, layer string) (string, error) {
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

func (p InstanceSegmentationPredictor) GetOutputLayerName(reader io.Reader, layer string) (string, error) {
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

func (p *InstanceSegmentationPredictor) runOptions() *proto.RunOptions {
	if p.TraceLevel() >= tracer.FRAMEWORK_TRACE {
		return &proto.RunOptions{
			TraceLevel: proto.RunOptions_SOFTWARE_TRACE,
		}
	}
	return nil
}

// Predict ...
func (p *InstanceSegmentationPredictor) Predict(ctx context.Context, data interface{}, opts ...options.Option) error {
	span, ctx := tracer.StartSpanFromContext(ctx, tracer.APPLICATION_TRACE, "predict")
	defer span.Finish()

	if data == nil {
		return errors.New("input data nil")
	}
	input, ok := data.([]*gotensor.Dense)
	if !ok {
		return errors.New("input data is not slice of dense tensors")
	}

	tensor, err := makeTensorFromGoTensors(input)
	if err != nil {
		return err
	}
	// TODO interface image enhancement call to predictor backend
	return nil
}

// ReadPredictedFeatures ...
func (p *InstanceSegmentationPredictor) ReadPredictedFeatures(ctx context.Context) ([]dlframework.Features, error) {
	span, ctx := tracer.StartSpanFromContext(ctx, tracer.APPLICATION_TRACE, "read_predicted_features")
	defer span.Finish()

	boxes := p.boxes.([][][]float32)
	probabilities := p.probabilities.([][]float32)
	classes := p.classes.([][]float32)
	masks := p.masks.([][][][]float32)

	return p.CreateInstanceSegmentFeatures(ctx, probabilities, classes, boxes, masks, p.labels)
}

func (p *InstanceSegmentationPredictor) Reset(ctx context.Context) error {

	return nil
}

func (p *InstanceSegmentationPredictor) Close() error {
	return nil
}

func (p InstanceSegmentationPredictor) Modality() (dlframework.Modality, error) {
	return dlframework.ImageInstanceSegmentationModality, nil
}

func init() {
	config.AfterInit(func() {
		framework := pytorch.FrameworkManifest
		agent.AddPredictor(framework, &InstanceSegmentationPredictor{
			ImagePredictor: common.ImagePredictor{
				Base: common.Base{
					Framework: framework,
				},
			},
		})
	})
}
