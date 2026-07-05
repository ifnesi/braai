package tools

import (
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
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

// coerceInt converts a JSON-decoded value to int. Local models frequently emit
// numbers as strings (e.g. "3"), so accept float64, int, json.Number, and
// numeric strings rather than only float64.
func coerceInt(v any) (int, bool) {
	switch n := v.(type) {
	case float64:
		return int(n), true
	case int:
		return n, true
	case json.Number:
		if i, err := n.Int64(); err == nil {
			return int(i), true
		}
	case string:
		if f, err := strconv.ParseFloat(strings.TrimSpace(n), 64); err == nil {
			return int(f), true
		}
	}
	return 0, false
}

// coerceBool converts a JSON-decoded value to bool, accepting real bools as well
// as the string/number forms small models tend to produce.
func coerceBool(v any) (bool, bool) {
	switch b := v.(type) {
	case bool:
		return b, true
	case float64:
		return b != 0, true
	case string:
		switch strings.ToLower(strings.TrimSpace(b)) {
		case "true", "yes", "1", "on":
			return true, true
		case "false", "no", "0", "off":
			return false, true
		}
	}
	return false, false
}

// coerceFloat converts a JSON-decoded value to float64, tolerating numeric
// strings the way coerceInt/coerceBool tolerate their forms.
func coerceFloat(v any) (float64, bool) {
	switch n := v.(type) {
	case float64:
		return n, true
	case int:
		return float64(n), true
	case json.Number:
		if f, err := n.Float64(); err == nil {
			return f, true
		}
	case string:
		if f, err := strconv.ParseFloat(strings.TrimSpace(n), 64); err == nil {
			return f, true
		}
	}
	return 0, false
}

// intArg extracts a required integer argument (tolerant of numeric strings).
func intArg(args map[string]any, key string) (int, error) {
	v, ok := args[key]
	if !ok {
		return 0, fmt.Errorf("missing required argument %q", key)
	}
	n, ok := coerceInt(v)
	if !ok {
		return 0, fmt.Errorf("argument %q must be an integer", key)
	}
	return n, nil
}

// optionalIntArg extracts an optional integer argument (tolerant of numeric strings).
func optionalIntArg(args map[string]any, key string, def int) int {
	if v, ok := args[key]; ok {
		if n, ok := coerceInt(v); ok {
			return n
		}
	}
	return def
}

// optionalBoolArg extracts an optional boolean argument (tolerant of "true"/"1"/etc.).
func optionalBoolArg(args map[string]any, key string, def bool) bool {
	if v, ok := args[key]; ok {
		if b, ok := coerceBool(v); ok {
			return b
		}
	}
	return def
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
