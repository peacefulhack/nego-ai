package adapters

import (
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestLinearLoRAForwardAndInitialization(t *testing.T) {
	l, err := NewLinearLoRA(2, 2, 2, 4, 42)
	if err != nil {
		t.Fatal(err)
	}
	other, err := NewLinearLoRA(2, 2, 2, 4, 42)
	if err != nil || !reflect.DeepEqual(l, other) {
		t.Fatal("seed must reproduce initialization")
	}
	x, base := []float32{2, -1}, []float32{1, 2}
	got, err := l.Forward(x, base)
	if err != nil || !reflect.DeepEqual(got, base) {
		t.Fatalf("new adapter = %v, %v", got, err)
	}
	l.A, l.B = []float32{1, 2, -1, .5}, []float32{3, 4, -2, 1}
	got, err = l.Forward(x, base)
	if err != nil || !reflect.DeepEqual(got, []float32{-19, -3}) {
		t.Fatalf("forward = %v, %v", got, err)
	}
	if !reflect.DeepEqual(x, []float32{2, -1}) || !reflect.DeepEqual(base, []float32{1, 2}) {
		t.Fatal("forward mutated input")
	}
}

func TestLinearLoRABackwardFiniteDifferences(t *testing.T) {
	l, err := NewLinearLoRA(3, 2, 2, 3, 7)
	if err != nil {
		t.Fatal(err)
	}
	l.B = []float32{.2, -.3, .4, .7}
	x, dy, base := []float32{.5, -.7, .3}, []float32{.8, -.6}, []float32{.1, .2}
	g, err := l.Backward(x, dy)
	if err != nil {
		t.Fatal(err)
	}
	objective := func() float64 {
		y, err := l.Forward(x, base)
		if err != nil {
			t.Fatal(err)
		}
		return float64(y[0])*float64(dy[0]) + float64(y[1])*float64(dy[1])
	}
	for _, tc := range []struct {
		name              string
		values, gradients []float32
	}{{"A", l.A, g.A}, {"B", l.B, g.B}, {"input", x, g.Input}} {
		for i, original := range tc.values {
			const h = float32(.001)
			tc.values[i] = original + h
			plus := objective()
			tc.values[i] = original - h
			minus := objective()
			tc.values[i] = original
			want := (plus - minus) / float64(2*h)
			if math.Abs(float64(tc.gradients[i])-want) > .0002 {
				t.Fatalf("%s[%d] analytic=%g finite difference=%g", tc.name, i, tc.gradients[i], want)
			}
		}
	}
}

func TestLinearLoRASGDLearnsAndReloads(t *testing.T) {
	l, err := NewLinearLoRA(2, 2, 1, 1, 42)
	if err != nil {
		t.Fatal(err)
	}
	x, base, target := []float32{1, .5}, []float32{.1, -.1}, []float32{2, -2}
	step := func(l *LinearLoRA) float64 {
		y, err := l.Forward(x, base)
		if err != nil {
			t.Fatal(err)
		}
		var loss float64
		dy := make([]float32, len(y))
		for i := range y {
			dy[i] = y[i] - target[i]
			loss += .5 * float64(dy[i]) * float64(dy[i])
		}
		g, err := l.Backward(x, dy)
		if err != nil {
			t.Fatal(err)
		}
		if err := l.StepSGD(g, .05); err != nil {
			t.Fatal(err)
		}
		return loss
	}
	initial := step(l)
	for i := 0; i < 99; i++ {
		step(l)
	}
	if loss := step(l); loss >= initial*.001 {
		t.Fatalf("loss did not decrease enough: initial=%g final=%g", initial, loss)
	}
	path := filepath.Join(t.TempDir(), "adapter.json")
	if err := SaveLinearLoRA(path, l); err != nil {
		t.Fatal(err)
	}
	reloaded, err := LoadLinearLoRA(path)
	if err != nil || !reflect.DeepEqual(l, reloaded) {
		t.Fatalf("reload = %v", err)
	}
	step(l)
	step(reloaded)
	if !reflect.DeepEqual(l, reloaded) {
		t.Fatal("SGD resume differs from uninterrupted update")
	}
	before, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := SaveLinearLoRA(path, l); err == nil {
		t.Fatal("must not overwrite checkpoint")
	}
	after, err := os.ReadFile(path)
	if err != nil || !reflect.DeepEqual(before, after) {
		t.Fatal("existing checkpoint changed")
	}
}

func TestLinearLoRARejectsInvalidInputsAndUpdates(t *testing.T) {
	for _, dims := range [][3]int{{0, 2, 1}, {2, -1, 1}, {2, 2, 0}, {2, 2, 3}, {int(^uint(0) >> 1), 2, 1}} {
		if _, err := NewLinearLoRA(dims[0], dims[1], dims[2], 1, 0); err == nil {
			t.Fatalf("accepted shape %v", dims)
		}
	}
	for _, alpha := range []float32{0, -1, float32(math.NaN()), float32(math.Inf(1))} {
		if _, err := NewLinearLoRA(2, 2, 1, alpha, 0); err == nil {
			t.Fatal("accepted invalid alpha")
		}
	}
	l, _ := NewLinearLoRA(2, 2, 1, 1, 42)
	for _, x := range [][]float32{nil, {1}, {1, float32(math.NaN())}} {
		if _, err := l.Forward(x, []float32{0, 0}); err == nil {
			t.Fatal("accepted invalid forward input")
		}
		if _, err := l.Backward(x, []float32{0, 0}); err == nil {
			t.Fatal("accepted invalid backward input")
		}
	}
	a, b := append([]float32(nil), l.A...), append([]float32(nil), l.B...)
	for _, g := range []LinearLoRAGradients{
		{},
		{A: []float32{1, 1}, B: []float32{1, float32(math.NaN())}},
		{A: []float32{1, 1}, B: []float32{math.MaxFloat32, math.MaxFloat32}},
	} {
		if err := l.StepSGD(g, 2); err == nil {
			t.Fatal("expected invalid update error")
		}
		if !reflect.DeepEqual(a, l.A) || !reflect.DeepEqual(b, l.B) {
			t.Fatal("failed update changed weights")
		}
	}
	l.A = nil
	if err := l.Validate(); err == nil {
		t.Fatal("accepted missing weights")
	}
	if _, err := (*LinearLoRA)(nil).Forward(nil, nil); err == nil {
		t.Fatal("accepted nil adapter")
	}
}

func TestLoadLinearLoRARejectsInvalidFiles(t *testing.T) {
	path := filepath.Join(t.TempDir(), "adapter.json")
	for _, content := range []string{`{`, `null`, `{}`, `{"version":1,"type":"token_bias"}`, `{"version":1,"type":"lora_linear","input_size":2,"output_size":2,"rank":1,"alpha":1,"a":[1],"b":[0,0]}`} {
		if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
			t.Fatal(err)
		}
		if _, err := LoadLinearLoRA(path); err == nil {
			t.Fatalf("accepted invalid file: %s", content)
		}
	}
	if err := os.Truncate(path, maxLoRAFileBytes+1); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadLinearLoRA(path); err == nil {
		t.Fatal("accepted oversized file")
	}
}
