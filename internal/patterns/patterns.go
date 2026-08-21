package patterns

import (
	"regexp"
	"strings"
)

func Match(repoPath string, include, exclude []string) bool {
	repoPath = strings.ReplaceAll(repoPath, "\\", "/")
	if len(include) > 0 && !anyMatch(repoPath, include) {
		return false
	}
	if len(exclude) > 0 && anyMatch(repoPath, exclude) {
		return false
	}
	return true
}

func anyMatch(repoPath string, patterns []string) bool {
	for _, pattern := range patterns {
		if ok, _ := regexp.MatchString(globRegex(pattern), repoPath); ok {
			return true
		}
	}
	return false
}

func globRegex(pattern string) string {
	pattern = strings.ReplaceAll(pattern, "\\", "/")
	var b strings.Builder
	b.WriteString("^")
	for i := 0; i < len(pattern); i++ {
		switch pattern[i] {
		case '*':
			b.WriteString(".*")
		case '?':
			b.WriteByte('.')
		case '.', '+', '(', ')', '|', '^', '$', '{', '}', '\\':
			b.WriteByte('\\')
			b.WriteByte(pattern[i])
		case '[':
			j := i + 1
			for j < len(pattern) && pattern[j] != ']' {
				j++
			}
			if j < len(pattern) {
				b.WriteString(pattern[i : j+1])
				i = j
			} else {
				b.WriteString(`\[`)
			}
		default:
			b.WriteByte(pattern[i])
		}
	}
	b.WriteString("$")
	return b.String()
}
