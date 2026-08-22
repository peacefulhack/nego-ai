package main

import (
	"encoding/json"
	"fmt"
	"log"
	"os"

	"github.com/gakon/nego-ai/training"
)

func main() {
	data, err := os.ReadFile("examples/5.train/train-job.json")
	if err != nil {
		log.Fatal(err)
	}
	var job training.JobSpec
	if err := json.Unmarshal(data, &job); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("job=%s command=%s args=%d work_dir=%s\n", job.Name, job.Command, len(job.Args), job.WorkDir)
}
