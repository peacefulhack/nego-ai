package hub

import (
	"errors"
	"fmt"

	"github.com/gakon/nego-ai/internal/hfhub"
)

type ErrorKind string

const (
	ErrInvalidOptions     ErrorKind = "invalid_options"
	ErrNotFound           ErrorKind = "not_found"
	ErrUnauthorized       ErrorKind = "unauthorized"
	ErrForbidden          ErrorKind = "forbidden"
	ErrGated              ErrorKind = "gated"
	ErrRateLimited        ErrorKind = "rate_limited"
	ErrRevisionNotFound   ErrorKind = "revision_not_found"
	ErrChecksumMismatch   ErrorKind = "checksum_mismatch"
	ErrIncompleteSnapshot ErrorKind = "incomplete_snapshot"
	ErrNetwork            ErrorKind = "network"
	ErrUnexpectedResponse ErrorKind = "unexpected_response"
)

type Error struct {
	Kind       ErrorKind
	Message    string
	StatusCode int
	URL        string
	Err        error
}

func (e *Error) Error() string {
	if e.Message != "" {
		return e.Message
	}
	if e.Err != nil {
		return e.Err.Error()
	}
	return string(e.Kind)
}

func (e *Error) Unwrap() error {
	return e.Err
}

func IsNotFound(err error) bool {
	var e *Error
	return errors.As(err, &e) && e.Kind == ErrNotFound
}

func IsAuth(err error) bool {
	var e *Error
	return errors.As(err, &e) && (e.Kind == ErrUnauthorized || e.Kind == ErrForbidden || e.Kind == ErrGated)
}

func invalidOptions(format string, args ...any) error {
	return &Error{Kind: ErrInvalidOptions, Message: fmt.Sprintf(format, args...)}
}

func notFound(format string, args ...any) error {
	return &Error{Kind: ErrNotFound, Message: fmt.Sprintf(format, args...)}
}

func incompleteSnapshot(format string, args ...any) error {
	return &Error{Kind: ErrIncompleteSnapshot, Message: fmt.Sprintf(format, args...)}
}

func wrapHFError(err error) error {
	if err == nil {
		return nil
	}
	var httpErr *hfhub.HTTPError
	if !errors.As(err, &httpErr) {
		return &Error{Kind: ErrNetwork, Message: err.Error(), Err: err}
	}

	kind := ErrUnexpectedResponse
	switch httpErr.StatusCode {
	case 401:
		kind = ErrUnauthorized
	case 403:
		if httpErr.MaybeGated {
			kind = ErrGated
		} else {
			kind = ErrForbidden
		}
	case 404:
		kind = ErrNotFound
	case 429:
		kind = ErrRateLimited
	}
	return &Error{
		Kind:       kind,
		Message:    httpErr.Error(),
		StatusCode: httpErr.StatusCode,
		URL:        httpErr.URL,
		Err:        err,
	}
}
