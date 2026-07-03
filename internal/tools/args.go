package tools

import (
	"encoding/json"
	"fmt"
)

func unknownToolError(name string) error {
	return fmt.Errorf("unknown tool: %s", name)
}

// stringArg extracts a required string argument.
func stringArg(args map[string]any, key string) (string, error) {
	v, ok := args[key]
	if !ok {
		return "", fmt.Errorf("missing required argument %q", key)
	}
	s, ok := v.(string)
	if !ok {
		return "", fmt.Errorf("argument %q must be a string", key)
	}
	return s, nil
}

// optionalStringArg extracts an optional string argument, returning def if absent.
func optionalStringArg(args map[string]any, key, def string) string {
	v, ok := args[key]
	if !ok {
		return def
	}
	s, ok := v.(string)
	if !ok {
		return def
	}
	return s
}

// intArg extracts a required integer argument (JSON numbers decode as float64).
func intArg(args map[string]any, key string) (int, error) {
	v, ok := args[key]
	if !ok {
		return 0, fmt.Errorf("missing required argument %q", key)
	}
	f, ok := v.(float64)
	if !ok {
		return 0, fmt.Errorf("argument %q must be an integer", key)
	}
	return int(f), nil
}

// optionalIntArg extracts an optional integer argument (JSON numbers decode as float64).
func optionalIntArg(args map[string]any, key string, def int) int {
	v, ok := args[key]
	if !ok {
		return def
	}
	f, ok := v.(float64)
	if !ok {
		return def
	}
	return int(f)
}

// optionalBoolArg extracts an optional boolean argument.
func optionalBoolArg(args map[string]any, key string, def bool) bool {
	v, ok := args[key]
	if !ok {
		return def
	}
	b, ok := v.(bool)
	if !ok {
		return def
	}
	return b
}

// stringSliceArg extracts an optional array-of-strings argument, ignoring
// non-string elements rather than failing the whole call. Returns nil if
// absent or not an array.
func stringSliceArg(args map[string]any, key string) []string {
	v, ok := args[key]
	if !ok {
		return nil
	}
	raw, ok := v.([]any)
	if !ok {
		return nil
	}
	out := make([]string, 0, len(raw))
	for _, item := range raw {
		if s, ok := item.(string); ok {
			out = append(out, s)
		}
	}
	return out
}

// marshalCompactJSON marshals a value to JSON without unnecessary whitespace.
// Used for token-efficient LLM responses. Returns a string and error.
func marshalCompactJSON(v any) (string, error) {
	b, err := json.Marshal(v)
	if err != nil {
		return "", err
	}
	return string(b), nil
}
