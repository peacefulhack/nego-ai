package native

import (
	"context"
	"sync"

	nego "github.com/gakon/nego-ai"
)

type nativeStream struct {
	tokens chan nego.Token
	done   chan struct{}
	cancel context.CancelFunc
	mu     sync.Mutex
	err    error
}

func newNativeStream(cancel context.CancelFunc) *nativeStream {
	return &nativeStream{
		tokens: make(chan nego.Token, 8),
		done:   make(chan struct{}),
		cancel: cancel,
	}
}

func (s *nativeStream) Tokens() <-chan nego.Token {
	return s.tokens
}

func (s *nativeStream) Err() error {
	<-s.done
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.err
}

func (s *nativeStream) Close() error {
	if s.cancel != nil {
		s.cancel()
	}
	<-s.done
	return nil
}

func (s *nativeStream) run(ctx context.Context, generate func(func(string) error) error) {
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

func (s *nativeStream) setErr(err error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.err == nil {
		s.err = err
	}
}
