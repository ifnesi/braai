package textextract

import (
	"archive/zip"
	"bytes"
	"encoding/csv"
	"encoding/json"
	"encoding/xml"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"unicode/utf8"

	"github.com/ledongthuc/pdf"
	xhtml "golang.org/x/net/html"
	"golang.org/x/net/html/atom"
)

const maxDecompressedBytes = 512 << 20 // 512 MiB, guards against zip bombs

// ExtractText auto-detects file format and extracts text from the file at the
// given path. The path must already be resolved and security-checked by the caller.
func ExtractText(absPath string) (string, error) {
	switch strings.ToLower(filepath.Ext(absPath)) {
	case ".txt", ".text", ".md", ".markdown", ".mdx", ".rst", ".adoc", ".asciidoc",
		".log", ".logs", ".out", ".err", ".diff", ".patch",
		".go", ".py", ".js", ".jsx", ".ts", ".tsx", ".c", ".h", ".cpp", ".cc",
		".hpp", ".java", ".kt", ".rs", ".rb", ".php", ".pl", ".lua", ".swift",
		".scala", ".clj", ".ex", ".exs", ".erl", ".hs", ".dart", ".r", ".jl",
		".cs", ".fs", ".sql",
		".sh", ".bash", ".zsh", ".fish", ".ps1", ".bat", ".cmd", ".awk", ".sed",
		".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".properties", ".env",
		".tf", ".tfvars", ".hcl",
		".css", ".scss", ".sass", ".less", ".graphql", ".gql":
		return readPlain(absPath)
	case ".rtf":
		return readRTF(absPath)
	case ".json", ".geojson", ".topojson", ".jsonld", ".har", ".ipynb",
		".map", ".webmanifest", ".arb", ".avsc", ".avpr", ".avro",
		".jsonc", ".json5", ".jsonl", ".ndjson":
		return readJSON(absPath)
	case ".xml":
		return readXML(absPath)
	case ".html", ".htm":
		return readHTML(absPath)
	case ".csv", ".tsv":
		return readCSV(absPath)
	case ".docx":
		return readOfficeParts(absPath, "word/document.xml")
	case ".xlsx":
		return readXlsx(absPath)
	case ".pptx":
		return readPptx(absPath)
	case ".pdf":
		return readPDF(absPath)
	default:
		// Fall back to plain text if the file looks like UTF-8 text.
		return readMaybeText(absPath)
	}
}

// --- Simple formats -------------------------------------------------------

func readPlain(path string) (string, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	return string(b), nil
}

func readMaybeText(path string) (string, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	if !utf8.Valid(b) || bytes.IndexByte(b, 0) != -1 {
		return "", fmt.Errorf("unsupported file type: %s", filepath.Ext(path))
	}
	return string(b), nil
}

func readJSON(path string) (string, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	var buf bytes.Buffer
	if err := json.Compact(&buf, b); err != nil {
		return string(b), nil // not valid JSON; return raw content
	}
	return buf.String(), nil
}

func readXML(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	return extractXMLText(f)
}

// blockAtoms are elements whose end should produce a line break.
var blockAtoms = map[atom.Atom]bool{
	atom.P: true, atom.Div: true, atom.Li: true, atom.Ul: true, atom.Ol: true,
	atom.Tr: true, atom.Table: true, atom.Section: true, atom.Article: true,
	atom.Header: true, atom.Footer: true, atom.Blockquote: true, atom.Pre: true,
	atom.H1: true, atom.H2: true, atom.H3: true, atom.H4: true, atom.H5: true, atom.H6: true,
	atom.Hr: true, atom.Dl: true, atom.Dd: true, atom.Dt: true, atom.Figure: true,
	atom.Nav: true, atom.Aside: true, atom.Main: true, atom.Address: true,
}

func readHTML(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()

	doc, err := xhtml.Parse(f)
	if err != nil {
		return "", err
	}

	var b strings.Builder
	var walk func(*xhtml.Node)
	walk = func(n *xhtml.Node) {
		if n.Type == xhtml.ElementNode {
			switch n.DataAtom {
			case atom.Script, atom.Style, atom.Head, atom.Noscript, atom.Template:
				return // skip these subtrees entirely
			case atom.Br:
				b.WriteByte('\n')
			}
		}
		if n.Type == xhtml.TextNode {
			b.WriteString(n.Data) // entities are already decoded
		}
		for c := n.FirstChild; c != nil; c = c.NextSibling {
			walk(c)
		}
		if n.Type == xhtml.ElementNode && blockAtoms[n.DataAtom] {
			b.WriteByte('\n')
		}
	}
	walk(doc)
	return NormalizeWhitespace(b.String()), nil
}

func readCSV(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()

	r := csv.NewReader(f)
	r.FieldsPerRecord = -1 // tolerate rows with differing column counts
	r.LazyQuotes = true    // tolerate stray quotes
	if strings.EqualFold(filepath.Ext(path), ".tsv") {
		r.Comma = '\t'
	}

	var b strings.Builder
	for {
		rec, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return "", err
		}
		b.WriteString(strings.Join(rec, "\t"))
		b.WriteByte('\n')
	}
	return b.String(), nil
}

// --- OOXML (docx / xlsx / pptx) ------------------------------------------

// readOfficeParts extracts run-aware text from the named part of an OOXML file.
func readOfficeParts(path string, part string) (string, error) {
	zr, err := zip.OpenReader(path)
	if err != nil {
		return "", err
	}
	defer zr.Close()

	for _, f := range zr.File {
		if f.Name == part {
			rc, err := openZip(f)
			if err != nil {
				return "", err
			}
			txt, err := extractOfficeText(rc)
			rc.Close()
			return txt, err
		}
	}
	return "", fmt.Errorf("%s not found", part)
}

func readPptx(path string) (string, error) {
	zr, err := zip.OpenReader(path)
	if err != nil {
		return "", err
	}
	defer zr.Close()

	var slides []*zip.File
	for _, f := range zr.File {
		if strings.HasPrefix(f.Name, "ppt/slides/slide") && strings.HasSuffix(f.Name, ".xml") {
			slides = append(slides, f)
		}
	}
	sortByNaturalName(slides)

	var out []string
	for _, f := range slides {
		rc, err := openZip(f)
		if err != nil {
			return "", err
		}
		txt, err := extractOfficeText(rc)
		rc.Close()
		if err != nil {
			return "", err
		}
		if strings.TrimSpace(txt) != "" {
			out = append(out, "# "+filepath.Base(f.Name), txt)
		}
	}
	return strings.Join(out, "\n\n"), nil
}

func readXlsx(path string) (string, error) {
	zr, err := zip.OpenReader(path)
	if err != nil {
		return "", err
	}
	defer zr.Close()

	files := make(map[string]*zip.File, len(zr.File))
	var sheets []*zip.File
	for _, f := range zr.File {
		files[f.Name] = f
		if strings.HasPrefix(f.Name, "xl/worksheets/") && strings.HasSuffix(f.Name, ".xml") {
			sheets = append(sheets, f)
		}
	}

	var shared []string
	if f, ok := files["xl/sharedStrings.xml"]; ok {
		if shared, err = parseSharedStrings(f); err != nil {
			return "", err
		}
	}
	sortByNaturalName(sheets)

	var out []string
	for _, f := range sheets {
		rows, err := parseWorksheet(f, shared)
		if err != nil {
			return "", err
		}
		if len(rows) > 0 {
			out = append(out, "# "+filepath.Base(f.Name))
			out = append(out, rows...)
		}
	}
	return strings.Join(out, "\n"), nil
}

func parseSharedStrings(f *zip.File) ([]string, error) {
	rc, err := openZip(f)
	if err != nil {
		return nil, err
	}
	defer rc.Close()

	dec := newXMLDecoder(rc)
	var values []string
	var buf strings.Builder
	inSI, capture, phonetic := false, false, false

	for {
		tok, err := dec.Token()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
		switch t := tok.(type) {
		case xml.StartElement:
			switch t.Name.Local {
			case "si":
				inSI, phonetic = true, false
				buf.Reset()
			case "rPh": // phonetic guide text – not real content
				phonetic = true
			case "t":
				capture = inSI && !phonetic
			}
		case xml.EndElement:
			switch t.Name.Local {
			case "si":
				values = append(values, buf.String())
				inSI = false
			case "rPh":
				phonetic = false
			case "t":
				capture = false
			}
		case xml.CharData:
			if capture {
				buf.Write(t)
			}
		}
	}
	return values, nil
}

func parseWorksheet(f *zip.File, shared []string) ([]string, error) {
	rc, err := openZip(f)
	if err != nil {
		return nil, err
	}
	defer rc.Close()

	dec := newXMLDecoder(rc)
	var rows []string

	var cells map[int]string // column index -> value for current row
	maxCol := -1
	cellType, cellRef := "", ""
	var val strings.Builder
	capture := false

	for {
		tok, err := dec.Token()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
		switch t := tok.(type) {
		case xml.StartElement:
			switch t.Name.Local {
			case "row":
				cells = make(map[int]string)
				maxCol = -1
			case "c":
				cellType, cellRef = "", ""
				for _, a := range t.Attr {
					switch a.Name.Local {
					case "t":
						cellType = a.Value
					case "r":
						cellRef = a.Value
					}
				}
			case "v", "t":
				capture = true
				val.Reset()
			}
		case xml.EndElement:
			switch t.Name.Local {
			case "v", "t":
				capture = false
			case "c":
				text := val.String()
				if cellType == "s" { // shared-string index
					if idx, e := strconv.Atoi(strings.TrimSpace(text)); e == nil && idx >= 0 && idx < len(shared) {
						text = shared[idx]
					}
				}
				if cells != nil {
					col := columnIndex(cellRef)
					if col < 0 {
						col = maxCol + 1 // fall back to positional
					}
					cells[col] = strings.TrimSpace(text)
					if col > maxCol {
						maxCol = col
					}
				}
			case "row":
				if line := joinRow(cells, maxCol); line != "" {
					rows = append(rows, line)
				}
				cells = nil
			}
		case xml.CharData:
			if capture {
				val.Write(t)
			}
		}
	}
	return rows, nil
}

func joinRow(cells map[int]string, maxCol int) string {
	if maxCol < 0 {
		return ""
	}
	cols := make([]string, maxCol+1)
	empty := true
	for i, v := range cells {
		if i >= 0 && i <= maxCol {
			cols[i] = v
			if v != "" {
				empty = false
			}
		}
	}
	if empty {
		return ""
	}
	return strings.Join(cols, "\t")
}

// columnIndex converts a cell reference like "C3" into a zero-based column index.
func columnIndex(ref string) int {
	col, n := 0, 0
	for _, c := range ref {
		switch {
		case c >= 'A' && c <= 'Z':
			col = col*26 + int(c-'A') + 1
		case c >= 'a' && c <= 'z':
			col = col*26 + int(c-'a') + 1
		default:
			goto done
		}
		n++
	}
done:
	if n == 0 {
		return -1
	}
	return col - 1
}

// --- PDF ------------------------------------------------------------------

func readPDF(path string) (string, error) {
	if bin, err := exec.LookPath("pdftotext"); err == nil {
		out, err := exec.Command(bin, "-layout", "-enc", "UTF-8", path, "-").Output()
		if err == nil && len(bytes.TrimSpace(out)) > 0 {
			return NormalizeWhitespace(string(out)), nil
		}
	}
	// Fall back to the ledongthuc/pdf implementation
	f, r, err := pdf.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()

	var out []string
	for i := 1; i <= r.NumPage(); i++ {
		txt, err := pdfPageText(r, i)
		if err != nil || strings.TrimSpace(txt) == "" {
			continue // skip blank or problematic pages rather than failing outright
		}
		out = append(out, txt)
	}
	if len(out) == 0 {
		return "", errors.New("no extractable text (PDF may be scanned/image-only or encrypted)")
	}
	return NormalizeWhitespace(strings.Join(out, "\n\n")), nil
}

// pdfPageText isolates each page and recovers from the panics the library can
// raise on malformed content streams, so one bad page won't crash extraction.
func pdfPageText(r *pdf.Reader, i int) (txt string, err error) {
	defer func() {
		if rec := recover(); rec != nil {
			err = fmt.Errorf("page %d: %v", i, rec)
		}
	}()
	p := r.Page(i)
	if p.V.IsNull() {
		return "", nil
	}
	return p.GetPlainText(nil)
}

// --- RTF ------------------------------------------------------------------

func readRTF(path string) (string, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	return NormalizeWhitespace(stripRTF(string(b))), nil
}

// RTF parsing is complex; we use a regex-based state machine.
// Adapted from classic RTF-to-text converters.
var rtfSpecial = map[string]string{
	"par": "\n", "sect": "\n", "page": "\n", "line": "\n", "tab": "\t",
	"emdash": "—", "endash": "–", "emspace": " ",
	"enspace": " ", "qmspace": " ", "bullet": "•",
	"lquote": "‘", "rquote": "’", "ldblquote": "“", "rdblquote": "”",
}

var rtfDestinations = makeSet(
	"aftncn", "aftnsep", "aftnsepc", "annotation", "atnauthor", "atndate", "atnicn",
	"atnid", "atnparent", "atnref", "atntime", "atrfend", "atrfstart", "author",
	"background", "bkmkend", "bkmkstart", "blipuid", "buptim", "category",
	"colorschememapping", "colortbl", "comment", "company", "creatim", "datafield",
	"datastore", "defchp", "defpap", "do", "doccomm", "docvar", "dptxbxtext",
	"ebcend", "ebcstart", "factoidname", "falt", "fchars", "ffdeftext", "ffentrymcr",
	"ffexitmcr", "ffformat", "ffhelptext", "ffl", "ffname", "ffstattext", "field",
	"file", "filetbl", "fldinst", "fldrslt", "fldtype", "fname", "fontemb",
	"fontfile", "fonttbl", "footer", "footerf", "footerl", "footerr", "footnote",
	"formfield", "ftncn", "ftnsep", "ftnsepc", "g", "generator", "gridtbl", "header",
	"headerf", "headerl", "headerr", "hl", "hlfr", "hlinkbase", "hlloc", "hlsrc",
	"hsv", "htmltag", "info", "keycode", "keywords", "latentstyles", "lchars",
	"levelnumbers", "leveltext", "lfolevel", "linkval", "list", "listlevel",
	"listname", "listoverride", "listoverridetable", "listpicture", "liststylename",
	"listtable", "listtext", "lsdlockedexcept",
)

func makeSet(words ...string) map[string]bool {
	m := make(map[string]bool, len(words))
	for _, w := range words {
		m[w] = true
	}
	return m
}

func stripRTF(text string) string {
	// Simplified RTF stripping: extract text while handling basic control sequences.
	var out strings.Builder
	ignorable := false
	var stack []bool

	i := 0
	for i < len(text) {
		switch text[i] {
		case '{':
			stack = append(stack, ignorable)
			i++
		case '}':
			if len(stack) > 0 {
				ignorable = stack[len(stack)-1]
				stack = stack[:len(stack)-1]
			}
			i++
		case '\\':
			i++
			if i >= len(text) {
				break
			}
			if text[i] == '\'' && i+2 < len(text) {
				// Hex escape
				hex := text[i+1 : i+3]
				if v, err := strconv.ParseInt(hex, 16, 8); err == nil && !ignorable {
					out.WriteRune(rune(v))
				}
				i += 3
			} else if text[i] >= 'a' && text[i] <= 'z' {
				// Control word
				start := i
				for i < len(text) && text[i] >= 'a' && text[i] <= 'z' {
					i++
				}
				word := text[start:i]
				if rtfDestinations[word] {
					ignorable = true
				} else if rtfSpecial[word] != "" && !ignorable {
					out.WriteString(rtfSpecial[word])
				}
				// Skip numeric arg if present
				if i < len(text) && (text[i] == '-' || (text[i] >= '0' && text[i] <= '9')) {
					for i < len(text) && text[i] >= '0' && text[i] <= '9' {
						i++
					}
					if i < len(text) && text[i] == ' ' {
						i++
					}
				}
			} else {
				// Single-char escape like \{ or \\
				switch text[i] {
				case '{', '}', '\\':
					if !ignorable {
						out.WriteByte(text[i])
					}
				case '~':
					if !ignorable {
						out.WriteRune(' ')
					}
				}
				i++
			}
		default:
			if !ignorable && text[i] > 31 {
				out.WriteByte(text[i])
			}
			i++
		}
	}
	return out.String()
}

// --- XML text helpers -----------------------------------------------------

// extractOfficeText pulls run-aware text from docx/pptx XML: text lives in <t>
// elements, paragraphs (<p>) and <br>/<cr> become newlines, <tab> becomes a tab.
func extractOfficeText(r io.Reader) (string, error) {
	dec := newXMLDecoder(r)
	var b strings.Builder
	depthT := 0

	for {
		tok, err := dec.Token()
		if err == io.EOF {
			break
		}
		if err != nil {
			return "", err
		}
		switch t := tok.(type) {
		case xml.StartElement:
			switch t.Name.Local {
			case "t":
				depthT++
			case "tab":
				b.WriteByte('\t')
			case "br", "cr":
				b.WriteByte('\n')
			}
		case xml.EndElement:
			switch t.Name.Local {
			case "t":
				if depthT > 0 {
					depthT--
				}
			case "p": // end of paragraph
				b.WriteByte('\n')
			}
		case xml.CharData:
			if depthT > 0 {
				b.Write(t)
			}
		}
	}
	return NormalizeWhitespace(b.String()), nil
}

// extractXMLText returns all character data from a generic XML document.
func extractXMLText(r io.Reader) (string, error) {
	dec := newXMLDecoder(r)
	var parts []string
	for {
		tok, err := dec.Token()
		if err == io.EOF {
			break
		}
		if err != nil {
			return "", err
		}
		if ch, ok := tok.(xml.CharData); ok {
			if s := strings.TrimSpace(string(ch)); s != "" {
				parts = append(parts, s)
			}
		}
	}
	return NormalizeWhitespace(strings.Join(parts, "\n")), nil
}

func newXMLDecoder(r io.Reader) *xml.Decoder {
	dec := xml.NewDecoder(r)
	dec.Strict = false          // tolerate minor malformations
	dec.Entity = xml.HTMLEntity // resolve HTML entities like &nbsp;
	dec.CharsetReader = func(_ string, in io.Reader) (io.Reader, error) {
		return in, nil // pass bytes through; OOXML is UTF-8
	}
	return dec
}

// --- Utilities ------------------------------------------------------------

// openZip opens a zip entry with a size cap to defend against decompression bombs.
func openZip(f *zip.File) (io.ReadCloser, error) {
	rc, err := f.Open()
	if err != nil {
		return nil, err
	}
	return &limitedReadCloser{
		r: io.LimitReader(rc, maxDecompressedBytes+1),
		c: rc,
	}, nil
}

type limitedReadCloser struct {
	r     io.Reader
	c     io.Closer
	total int64
}

func (l *limitedReadCloser) Read(p []byte) (int, error) {
	n, err := l.r.Read(p)
	l.total += int64(n)
	if l.total > maxDecompressedBytes {
		return n, errors.New("decompressed size limit exceeded")
	}
	return n, err
}

func (l *limitedReadCloser) Close() error { return l.c.Close() }

// sortByNaturalName sorts zip files so slide2.xml precedes slide10.xml.
func sortByNaturalName(files []*zip.File) {
	// Quicksort by natural order (numeric sequences treated as numbers)
	quickSort(files, 0, len(files)-1)
}

func quickSort(files []*zip.File, lo, hi int) {
	if lo < hi {
		p := partition(files, lo, hi)
		quickSort(files, lo, p-1)
		quickSort(files, p+1, hi)
	}
}

func partition(files []*zip.File, lo, hi int) int {
	pivot := files[hi].Name
	i := lo - 1
	for j := lo; j < hi; j++ {
		if naturalLess(files[j].Name, pivot) {
			i++
			files[i], files[j] = files[j], files[i]
		}
	}
	files[i+1], files[hi] = files[hi], files[i+1]
	return i + 1
}

func naturalLess(a, b string) bool {
	i, j := 0, 0
	for i < len(a) && j < len(b) {
		if isDigit(a[i]) && isDigit(b[j]) {
			si, sj := i, j
			for i < len(a) && isDigit(a[i]) {
				i++
			}
			for j < len(b) && isDigit(b[j]) {
				j++
			}
			na, _ := strconv.Atoi(a[si:i])
			nb, _ := strconv.Atoi(b[sj:j])
			if na != nb {
				return na < nb
			}
			continue
		}
		if a[i] != b[j] {
			return a[i] < b[j]
		}
		i++
		j++
	}
	return len(a)-i < len(b)-j
}

func isDigit(c byte) bool { return c >= '0' && c <= '9' }
