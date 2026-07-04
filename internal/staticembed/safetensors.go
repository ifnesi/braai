package staticembed

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
)

// readSafetensors extracts the named 2-D tensor (falling back to the first
// tensor) from a .safetensors file as a flat []float32 plus its [rows, cols]
// shape. Supports F32 and F16 dtypes.
func readSafetensors(path, name string) ([]float32, int, int, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, 0, 0, err
	}
	defer f.Close()

	var hlen uint64
	if err := binary.Read(f, binary.LittleEndian, &hlen); err != nil {
		return nil, 0, 0, err
	}
	header := make([]byte, hlen)
	if _, err := io.ReadFull(f, header); err != nil {
		return nil, 0, 0, err
	}
	var meta map[string]json.RawMessage
	if err := json.Unmarshal(header, &meta); err != nil {
		return nil, 0, 0, err
	}

	raw, ok := meta[name]
	if !ok {
		for k, v := range meta {
			if k == "__metadata__" {
				continue
			}
			raw = v
			ok = true
			break
		}
	}
	if !ok {
		return nil, 0, 0, fmt.Errorf("no tensor found in %s", path)
	}

	var td struct {
		Dtype   string  `json:"dtype"`
		Shape   []int   `json:"shape"`
		Offsets []int64 `json:"data_offsets"`
	}
	if err := json.Unmarshal(raw, &td); err != nil {
		return nil, 0, 0, err
	}
	if len(td.Shape) != 2 {
		return nil, 0, 0, fmt.Errorf("expected 2-D embeddings, got shape %v", td.Shape)
	}
	if len(td.Offsets) != 2 {
		return nil, 0, 0, fmt.Errorf("bad data_offsets for tensor %q", name)
	}

	buf := make([]byte, td.Offsets[1]-td.Offsets[0])
	if _, err := f.Seek(8+int64(hlen)+td.Offsets[0], io.SeekStart); err != nil {
		return nil, 0, 0, err
	}
	if _, err := io.ReadFull(f, buf); err != nil {
		return nil, 0, 0, err
	}

	var data []float32
	switch td.Dtype {
	case "F32":
		data = make([]float32, len(buf)/4)
		for i := range data {
			data[i] = math.Float32frombits(binary.LittleEndian.Uint32(buf[i*4:]))
		}
	case "F16":
		data = make([]float32, len(buf)/2)
		for i := range data {
			data[i] = f16to32(binary.LittleEndian.Uint16(buf[i*2:]))
		}
	default:
		return nil, 0, 0, fmt.Errorf("unsupported dtype %q (want F32 or F16)", td.Dtype)
	}
	if len(data) != td.Shape[0]*td.Shape[1] {
		return nil, 0, 0, fmt.Errorf("tensor size mismatch: %d values for shape %v", len(data), td.Shape)
	}
	return data, td.Shape[0], td.Shape[1], nil
}

func f16to32(h uint16) float32 {
	sign := uint32(h&0x8000) << 16
	exp := uint32(h>>10) & 0x1f
	mant := uint32(h & 0x3ff)
	switch {
	case exp == 0:
		if mant == 0 {
			return math.Float32frombits(sign)
		}
		for mant&0x400 == 0 {
			mant <<= 1
			exp--
		}
		exp++
		mant &= 0x3ff
		return math.Float32frombits(sign | (exp+112)<<23 | mant<<13)
	case exp == 0x1f:
		return math.Float32frombits(sign | 0x7f800000 | mant<<13)
	default:
		return math.Float32frombits(sign | (exp+112)<<23 | mant<<13)
	}
}
