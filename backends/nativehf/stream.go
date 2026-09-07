package nativehf

import "github.com/gakon/nego-ai"

type staticStream struct {
	tokens chan nego.Token
	err    error
}

func newStaticStream(text string) *staticStream {
	ch := make(chan nego.Token, 1)
	ch <- nego.Token{Text: text}
	close(ch)
	return &staticStream{tokens: ch}
}

func (s *staticStream) Tokens() <-chan nego.Token {
	return s.tokens
}

func (s *staticStream) Err() error {
	if s == nil {
		return nil
	}
	return s.err
}

func (s *staticStream) Close() error {
	return nil
}
