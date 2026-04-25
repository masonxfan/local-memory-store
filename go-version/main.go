package main

import (
	"database/sql"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	_ "github.com/mattn/go-sqlite3"
	ort "github.com/yalue/onnxruntime_go"
)

// ============================================================
// Tokenizer — 简化版 WordPiece tokenizer for MiniLM
// ============================================================

type Tokenizer struct {
	Vocab   map[string]int64
	IDToStr map[int64]string
}

func LoadTokenizer(vocabPath string) (*Tokenizer, error) {
	data, err := os.ReadFile(vocabPath)
	if err != nil {
		return nil, fmt.Errorf("read vocab: %w", err)
	}
	vocab := make(map[string]int64)
	idToStr := make(map[int64]string)
	for i, line := range strings.Split(strings.TrimSpace(string(data)), "\n") {
		token := strings.TrimSpace(line)
		vocab[token] = int64(i)
		idToStr[int64(i)] = token
	}
	return &Tokenizer{Vocab: vocab, IDToStr: idToStr}, nil
}

func (t *Tokenizer) Encode(text string, maxLen int) (ids, mask, typeIDs []int64) {
	// Simple whitespace + WordPiece tokenization
	tokens := []int64{t.Vocab["[CLS]"]}
	words := strings.Fields(strings.ToLower(text))
	for _, word := range words {
		// Try full word first
		if id, ok := t.Vocab[word]; ok {
			tokens = append(tokens, id)
			continue
		}
		// WordPiece: split into subwords
		remaining := word
		for len(remaining) > 0 {
			found := false
			for end := len(remaining); end > 0; end-- {
				sub := remaining[:end]
				if len(remaining) != len(word) || end != len(remaining) {
					// Not the first subword, try with ##
				}
				var lookupKey string
				if remaining == word[:len(remaining)] && len(tokens) > 1 && remaining != word {
					lookupKey = "##" + sub
				} else if remaining != word {
					lookupKey = "##" + sub
				} else {
					lookupKey = sub
				}
				if id, ok := t.Vocab[lookupKey]; ok {
					tokens = append(tokens, id)
					remaining = remaining[end:]
					found = true
					break
				}
				// Also try without ## prefix
				if id, ok := t.Vocab[sub]; ok {
					tokens = append(tokens, id)
					remaining = remaining[end:]
					found = true
					break
				}
			}
			if !found {
				// Unknown token
				if id, ok := t.Vocab["[UNK]"]; ok {
					tokens = append(tokens, id)
				}
				break
			}
		}
	}
	tokens = append(tokens, t.Vocab["[SEP]"])

	// Pad or truncate
	if len(tokens) > maxLen {
		tokens = tokens[:maxLen]
		tokens[maxLen-1] = t.Vocab["[SEP]"]
	}

	ids = make([]int64, maxLen)
	mask = make([]int64, maxLen)
	typeIDs = make([]int64, maxLen)
	for i := 0; i < len(tokens); i++ {
		ids[i] = tokens[i]
		mask[i] = 1
	}
	return
}

// ============================================================
// Embedder — ONNX Runtime wrapper
// ============================================================

type Embedder struct {
	tokenizer *Tokenizer
	modelPath string
	maxLen    int
}

func NewEmbedder(modelDir string) (*Embedder, error) {
	// Initialize ONNX Runtime
	libPath := os.Getenv("ONNXRUNTIME_LIB_PATH")
	if libPath == "" {
		// Try common locations
		candidates := []string{
			"onnxruntime.dll",
			"libonnxruntime.so",
			"libonnxruntime.dylib",
			"/usr/lib/libonnxruntime.so",
			"/usr/local/lib/libonnxruntime.so",
			filepath.Join(modelDir, "libonnxruntime.so"),
			filepath.Join(modelDir, "onnxruntime.dll"),
		}
		for _, c := range candidates {
			if _, err := os.Stat(c); err == nil {
				libPath = c
				break
			}
		}
		if libPath == "" {
			return nil, fmt.Errorf("ONNX Runtime library not found. Set ONNXRUNTIME_LIB_PATH or place it in model dir")
		}
	}
	ort.SetSharedLibraryPath(libPath)
	if err := ort.InitializeEnvironment(); err != nil {
		return nil, fmt.Errorf("init ONNX Runtime: %w", err)
	}

	vocabPath := filepath.Join(modelDir, "vocab.txt")
	tokenizer, err := LoadTokenizer(vocabPath)
	if err != nil {
		return nil, err
	}

	modelPath := filepath.Join(modelDir, "model.onnx")
	if _, err := os.Stat(modelPath); err != nil {
		return nil, fmt.Errorf("model not found: %s", modelPath)
	}

	return &Embedder{
		tokenizer: tokenizer,
		modelPath: modelPath,
		maxLen:    128,
	}, nil
}

func (e *Embedder) Embed(text string) ([]float32, error) {
	ids, mask, typeIDs := e.tokenizer.Encode(text, e.maxLen)

	inputShape := ort.NewShape(1, int64(e.maxLen))

	inputIDs, err := ort.NewTensor(inputShape, ids)
	if err != nil {
		return nil, fmt.Errorf("create input_ids tensor: %w", err)
	}
	defer inputIDs.Destroy()

	attentionMask, err := ort.NewTensor(inputShape, mask)
	if err != nil {
		return nil, fmt.Errorf("create attention_mask tensor: %w", err)
	}
	defer attentionMask.Destroy()

	tokenTypeIDs, err := ort.NewTensor(inputShape, typeIDs)
	if err != nil {
		return nil, fmt.Errorf("create token_type_ids tensor: %w", err)
	}
	defer tokenTypeIDs.Destroy()

	// Output: [1, seq_len, 384]
	outputShape := ort.NewShape(1, int64(e.maxLen), 384)
	output, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		return nil, fmt.Errorf("create output tensor: %w", err)
	}
	defer output.Destroy()

	session, err := ort.NewAdvancedSession(
		e.modelPath,
		[]string{"input_ids", "attention_mask", "token_type_ids"},
		[]string{"last_hidden_state"},
		[]ort.ArbitraryTensor{inputIDs, attentionMask, tokenTypeIDs},
		[]ort.ArbitraryTensor{output},
		nil,
	)
	if err != nil {
		return nil, fmt.Errorf("create session: %w", err)
	}
	defer session.Destroy()

	if err := session.Run(); err != nil {
		return nil, fmt.Errorf("run inference: %w", err)
	}

	// Mean pooling over non-padding tokens
	outputData := output.GetData()
	dim := 384
	tokenCount := 0
	for i := 0; i < e.maxLen; i++ {
		if mask[i] == 1 {
			tokenCount++
		}
	}

	embedding := make([]float32, dim)
	for i := 0; i < e.maxLen; i++ {
		if mask[i] == 0 {
			continue
		}
		for j := 0; j < dim; j++ {
			embedding[j] += outputData[i*dim+j]
		}
	}
	for j := 0; j < dim; j++ {
		embedding[j] /= float32(tokenCount)
	}

	// L2 normalize
	var norm float64
	for _, v := range embedding {
		norm += float64(v) * float64(v)
	}
	norm = math.Sqrt(norm)
	if norm > 0 {
		for j := range embedding {
			embedding[j] = float32(float64(embedding[j]) / norm)
		}
	}

	return embedding, nil
}

func (e *Embedder) Close() {
	ort.DestroyEnvironment()
}

// ============================================================
// VectorStore — SQLite-backed vector storage
// ============================================================

type VectorStore struct {
	db *sql.DB
}

type Memory struct {
	ID        string            `json:"id"`
	Text      string            `json:"text"`
	User      string            `json:"user"`
	Metadata  map[string]string `json:"metadata,omitempty"`
	Timestamp string            `json:"timestamp"`
	Score     float64           `json:"score,omitempty"`
}

func NewVectorStore(dbPath string) (*VectorStore, error) {
	db, err := sql.Open("sqlite3", dbPath)
	if err != nil {
		return nil, err
	}

	_, err = db.Exec(`
		CREATE TABLE IF NOT EXISTS memories (
			id TEXT PRIMARY KEY,
			user TEXT NOT NULL,
			text TEXT NOT NULL,
			embedding BLOB NOT NULL,
			metadata TEXT DEFAULT '{}',
			timestamp TEXT NOT NULL,
			epoch INTEGER NOT NULL
		);
		CREATE INDEX IF NOT EXISTS idx_user ON memories(user);
	`)
	if err != nil {
		return nil, err
	}

	return &VectorStore{db: db}, nil
}

func (vs *VectorStore) Add(id, user, text string, embedding []float32, metadata map[string]string) error {
	embBytes := float32ToBytes(embedding)
	metaJSON, _ := json.Marshal(metadata)
	now := time.Now().UTC()

	_, err := vs.db.Exec(
		`INSERT INTO memories (id, user, text, embedding, metadata, timestamp, epoch) VALUES (?, ?, ?, ?, ?, ?, ?)`,
		id, user, text, embBytes, string(metaJSON), now.Format(time.RFC3339), now.Unix(),
	)
	return err
}

func (vs *VectorStore) Search(user, queryEmbeddingBytes []byte, queryEmbedding []float32, limit int) ([]Memory, error) {
	var rows *sql.Rows
	var err error

	if user != "" {
		rows, err = vs.db.Query(
			`SELECT id, user, text, embedding, metadata, timestamp FROM memories WHERE user = ?`, user)
	} else {
		rows, err = vs.db.Query(
			`SELECT id, user, text, embedding, metadata, timestamp FROM memories`)
	}
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	type scored struct {
		mem   Memory
		score float64
	}
	var results []scored

	for rows.Next() {
		var id, u, text, ts string
		var embBytes []byte
		var metaJSON string

		if err := rows.Scan(&id, &u, &text, &embBytes, &metaJSON, &ts); err != nil {
			continue
		}

		emb := bytesToFloat32(embBytes)
		score := cosineSimilarity(queryEmbedding, emb)

		var meta map[string]string
		json.Unmarshal([]byte(metaJSON), &meta)

		results = append(results, scored{
			mem:   Memory{ID: id, User: u, Text: text, Metadata: meta, Timestamp: ts},
			score: score,
		})
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].score > results[j].score
	})

	if len(results) > limit {
		results = results[:limit]
	}

	var memories []Memory
	for _, r := range results {
		r.mem.Score = math.Round(r.score*10000) / 10000
		memories = append(memories, r.mem)
	}
	return memories, nil
}

func (vs *VectorStore) Count(user string) int {
	var count int
	if user != "" {
		vs.db.QueryRow(`SELECT COUNT(*) FROM memories WHERE user = ?`, user).Scan(&count)
	} else {
		vs.db.QueryRow(`SELECT COUNT(*) FROM memories`).Scan(&count)
	}
	return count
}

func (vs *VectorStore) Close() {
	vs.db.Close()
}

// ============================================================
// Helpers
// ============================================================

func float32ToBytes(f []float32) []byte {
	buf := make([]byte, len(f)*4)
	for i, v := range f {
		bits := math.Float32bits(v)
		buf[i*4] = byte(bits)
		buf[i*4+1] = byte(bits >> 8)
		buf[i*4+2] = byte(bits >> 16)
		buf[i*4+3] = byte(bits >> 24)
	}
	return buf
}

func bytesToFloat32(b []byte) []float32 {
	f := make([]float32, len(b)/4)
	for i := range f {
		bits := uint32(b[i*4]) | uint32(b[i*4+1])<<8 | uint32(b[i*4+2])<<16 | uint32(b[i*4+3])<<24
		f[i] = math.Float32frombits(bits)
	}
	return f
}

func cosineSimilarity(a, b []float32) float64 {
	if len(a) != len(b) {
		return 0
	}
	var dot, normA, normB float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}

func generateID() string {
	return fmt.Sprintf("%d-%d", time.Now().UnixNano(), os.Getpid())
}

// ============================================================
// CLI
// ============================================================

func main() {
	if len(os.Args) < 2 {
		printUsage()
		os.Exit(1)
	}

	command := os.Args[1]
	exePath, _ := os.Executable()
	baseDir := filepath.Dir(exePath)
	dbPath := filepath.Join(baseDir, "memories.db")
	modelDir := filepath.Join(baseDir, "models")

	// Allow override via env
	if v := os.Getenv("MEMORY_DB_PATH"); v != "" {
		dbPath = v
	}
	if v := os.Getenv("MEMORY_MODEL_DIR"); v != "" {
		modelDir = v
	}

	switch command {
	case "add":
		fs := flag.NewFlagSet("add", flag.ExitOnError)
		user := fs.String("user", "", "User ID (required)")
		text := fs.String("text", "", "Memory text (required)")
		meta := fs.String("metadata", "{}", "JSON metadata")
		fs.Parse(os.Args[2:])

		if *user == "" || *text == "" {
			fmt.Println(`{"error": "--user and --text are required"}`)
			os.Exit(1)
		}

		embedder, err := NewEmbedder(modelDir)
		if err != nil {
			fmt.Printf(`{"error": "init embedder: %s"}`+"\n", err)
			os.Exit(1)
		}
		defer embedder.Close()

		embedding, err := embedder.Embed(*text)
		if err != nil {
			fmt.Printf(`{"error": "embedding: %s"}`+"\n", err)
			os.Exit(1)
		}

		store, err := NewVectorStore(dbPath)
		if err != nil {
			fmt.Printf(`{"error": "open db: %s"}`+"\n", err)
			os.Exit(1)
		}
		defer store.Close()

		var metadata map[string]string
		json.Unmarshal([]byte(*meta), &metadata)

		id := generateID()
		if err := store.Add(id, *user, *text, embedding, metadata); err != nil {
			fmt.Printf(`{"error": "store: %s"}`+"\n", err)
			os.Exit(1)
		}

		fmt.Printf(`{"status": "ok", "id": "%s", "count": %d}`+"\n", id, store.Count(""))

	case "search":
		fs := flag.NewFlagSet("search", flag.ExitOnError)
		user := fs.String("user", "", "Filter by user")
		query := fs.String("query", "", "Search query (required)")
		limit := fs.Int("limit", 3, "Max results")
		fs.Parse(os.Args[2:])

		if *query == "" {
			fmt.Println(`{"error": "--query is required"}`)
			os.Exit(1)
		}

		embedder, err := NewEmbedder(modelDir)
		if err != nil {
			fmt.Printf(`{"error": "init embedder: %s"}`+"\n", err)
			os.Exit(1)
		}
		defer embedder.Close()

		embedding, err := embedder.Embed(*query)
		if err != nil {
			fmt.Printf(`{"error": "embedding: %s"}`+"\n", err)
			os.Exit(1)
		}

		store, err := NewVectorStore(dbPath)
		if err != nil {
			fmt.Printf(`{"error": "open db: %s"}`+"\n", err)
			os.Exit(1)
		}
		defer store.Close()

		results, err := store.Search(*user, float32ToBytes(embedding), embedding, *limit)
		if err != nil {
			fmt.Printf(`{"error": "search: %s"}`+"\n", err)
			os.Exit(1)
		}

		out, _ := json.Marshal(map[string]any{
			"results": results,
			"total":   store.Count(*user),
		})
		fmt.Println(string(out))

	case "stats":
		store, err := NewVectorStore(dbPath)
		if err != nil {
			fmt.Printf(`{"error": "open db: %s"}`+"\n", err)
			os.Exit(1)
		}
		defer store.Close()

		fmt.Printf(`{"total_memories": %d}`+"\n", store.Count(""))

	default:
		printUsage()
		os.Exit(1)
	}
}

func printUsage() {
	fmt.Println(`Local Memory Store (Go + ONNX Runtime)

Usage:
  memory-store add    --user <name> --text <text> [--metadata <json>]
  memory-store search --query <text> [--user <name>] [--limit N]
  memory-store stats

Environment:
  ONNXRUNTIME_LIB_PATH  Path to ONNX Runtime shared library
  MEMORY_DB_PATH         Path to SQLite database (default: ./memories.db)
  MEMORY_MODEL_DIR       Path to model directory (default: ./models/)

Model directory should contain:
  models/
  ├── model.onnx     (all-MiniLM-L6-v2 ONNX model)
  └── vocab.txt      (WordPiece vocabulary)`)
}
