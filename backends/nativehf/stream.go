package nativehf

import (
	"context"
	"sync"

	nego "github.com/gakon/nego-ai"
)

type nativeHFStream struct {
	tokens chan nego.Token
	done   chan struct{}
	cancel context.CancelFunc
	mu     sync.Mutex
	err    error
}

func newNativeHFStream(cancel context.CancelFunc) *nativeHFStream {
	return &nativeHFStream{
		tokens: make(chan nego.Token, 8),
		done:   make(chan struct{}),
		cancel: cancel,
	}
}

func (s *nativeHFStream) Tokens() <-chan nego.Token {
	return s.tokens
}

func (s *nativeHFStream) Err() error {
	<-s.done
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.err
}

func (s *nativeHFStream) Close() error {
	if s.cancel != nil {
		s.cancel()
	}
	<-s.done
	return nil
}

func (s *nativeHFStream) run(ctx context.Context, generate func(func(string) error) error) {
	defer close(s.tokens)
	defer close(s.done)
	defer s.cancel()

	err := generate(func(text string) error {
		select {
		case s.tokens <- nego.Token{Text: text}:
			return nil
		case <-ctx.Done():
			return ctx.Err()
		}
	})
	if err != nil && ctx.Err() == nil {
		s.setErr(err)
	}
}

func (s *nativeHFStream) setErr(err error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.err == nil {
		s.err = err
	}
}
