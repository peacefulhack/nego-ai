package main

import (
	"fmt"
	"log"
	"os"
	"path/filepath"

	"github.com/gakon/nego-ai/share"
)

func main() {
	modelDir := "./outputs/qwen3-token-bias"
	if len(os.Args) > 2 {
		log.Fatal("usage: go run ./examples/8.share-trained-model [trained-output-dir]")
	}
	if len(os.Args) == 2 {
		modelDir = os.Args[1]
	}
	modelDir, err := filepath.Abs(modelDir)
	if err != nil {
		log.Fatal(err)
	}
	manifest, err := share.BuildManifest(share.ManifestOptions{Path: modelDir})
	if err != nil {
		log.Fatalf("inspect trained output: %v; run step 5 first: go run ./examples/5.train/native_adapter", err)
	}
	if manifest.NativeAdapter == nil {
		log.Fatal("expected a native adapter output directory from step 5")
	}
	archivePath := modelDir + ".tar.gz"
	if _, err := os.Lstat(archivePath); !os.IsNotExist(err) {
		log.Fatalf("archive %s already exists or cannot be inspected; use a new output directory for another training run", archivePath)
	}
	manifest, err = share.PackageArchive(share.PackageOptions{
		Path:            modelDir,
		Output:          archivePath,
		Gzip:            true,
		IncludeManifest: true,
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Packaged adapter: %s\n", archivePath)
	fmt.Printf("Base model: %s\n", manifest.BaseModel)
	fmt.Printf("Method: %s\n", manifest.NativeAdapter.Method)
	fmt.Printf("Files: %d plus nego-share-manifest.json\n", len(manifest.Files))
	fmt.Println("The archive contains the adapter, not the base model weights.")
	fmt.Println("Recipients need the same base model and should pass the extracted adapter.json with --adapter.")
}
