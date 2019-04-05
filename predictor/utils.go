package predictor

import (
	"image"
	"image/png"
	"os"
	"runtime"
	"runtime/debug"

	imagetypes "github.com/rai-project/image/types"
	gotensor "gorgonia.org/tensor"
)

func makeFloatSliceFromGoTensors(in0 []*gotensor.Dense) ([]float32, error) {
	return nil, nil
}

func toPng(filePath string, imgByte []byte, bounds image.Rectangle) {

	img := imagetypes.NewRGBImage(bounds)
	copy(img.Pix, imgByte)

	out, _ := os.Create(filePath)
	defer out.Close()

	err := png.Encode(out, img.ToRGBAImage())
	if err != nil {
		log.Println(err)
	}
}

func zeros(height, width, channels int) [][][]float32 {
	rows := make([][][]float32, height)
	for ii := range rows {
		columns := make([][]float32, width)
		for jj := range columns {
			columns[jj] = make([]float32, channels)
		}
		rows[ii] = columns
	}
	return rows
}

func forceGC() {
	runtime.GC()
	debug.FreeOSMemory()
}
