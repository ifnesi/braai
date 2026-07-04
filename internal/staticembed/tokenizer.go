package staticembed

import (
	"encoding/json"
	"os"
	"strings"
)

type tokenizer struct {
	vocab     map[string]int
	unk       string
	prefix    string
	maxChars  int
	lowercase bool
}

func loadTokenizer(path string) (*tokenizer, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var raw struct {
		Normalizer json.RawMessage `json:"normalizer"`
		Model      struct {
			Vocab                   map[string]int `json:"vocab"`
			UnkToken                string         `json:"unk_token"`
			ContinuingSubwordPrefix string         `json:"continuing_subword_prefix"`
			MaxInputCharsPerWord    int            `json:"max_input_chars_per_word"`
		} `json:"model"`
	}
	if err := json.Unmarshal(b, &raw); err != nil {
		return nil, err
	}
	tk := &tokenizer{
		vocab:     raw.Model.Vocab,
		unk:       raw.Model.UnkToken,
		prefix:    raw.Model.ContinuingSubwordPrefix,
		maxChars:  raw.Model.MaxInputCharsPerWord,
		lowercase: true, // BertNormalizer default; overridden below if explicit
	}
	if tk.unk == "" {
		tk.unk = "[UNK]"
	}
	if tk.prefix == "" {
		tk.prefix = "##"
	}
	if tk.maxChars == 0 {
		tk.maxChars = 100
	}
	// Respect an explicit lowercase:false in the BERT normalizer (cased models).
	if strings.Contains(string(raw.Normalizer), `"lowercase":false`) {
		tk.lowercase = false
	}
	return tk, nil
}
