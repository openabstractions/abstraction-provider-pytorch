package torch

import (
	"archive/zip"
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/base64"
	"encoding/csv"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"strconv"
	"strings"

	identity "github.com/openabstractions/abstraction-identity"
	"github.com/openabstractions/abstraction-provider-kit/go/sandbox"
)

// A lockEntry is one distribution of a --require-hashes lock: its name and
// version as written, and the SHA-256 of each wheel it admits.
type lockEntry struct {
	Name, Version string
	Hashes        []string
}

// parseLock reads a pip lock of the stack review's shape: name==version
// with a continuation line per --hash=sha256:..., comments and blank lines
// ignored. Every entry needs at least one hash and an exact version.
func parseLock(data []byte) ([]lockEntry, error) {
	var logical []string
	var current strings.Builder
	scanner := bufio.NewScanner(bytes.NewReader(data))
	for scanner.Scan() {
		line := scanner.Text()
		if i := strings.Index(line, "#"); i >= 0 {
			line = line[:i]
		}
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		cont := strings.HasSuffix(line, `\`)
		current.WriteString(strings.TrimSpace(strings.TrimSuffix(line, `\`)) + " ")
		if !cont {
			logical = append(logical, strings.TrimSpace(current.String()))
			current.Reset()
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	if s := strings.TrimSpace(current.String()); s != "" {
		logical = append(logical, s)
	}
	var out []lockEntry
	for _, l := range logical {
		fields := strings.Fields(l)
		name, version, ok := strings.Cut(fields[0], "==")
		if !ok || name == "" || version == "" || strings.ContainsAny(version, "*<>=!~") {
			return nil, fmt.Errorf("the lock line %q pins no exact version", l)
		}
		e := lockEntry{Name: name, Version: version}
		for _, f := range fields[1:] {
			digest, ok := strings.CutPrefix(f, "--hash=sha256:")
			if !ok || !lowerHex(digest, 64) {
				return nil, fmt.Errorf("the lock line %q has %q, not --hash=sha256:<hex>", l, f)
			}
			e.Hashes = append(e.Hashes, digest)
		}
		if len(e.Hashes) == 0 {
			return nil, fmt.Errorf("the lock line %q has no hash", l)
		}
		out = append(out, e)
	}
	if len(out) == 0 {
		return nil, errors.New("the lock names no distribution")
	}
	return out, nil
}

// wheelsByDigest finds every .whl file in folders and below, keyed by its
// SHA-256: a lock entry is found by its hash alone, never by its name.
// Links and junctions are not followed.
func wheelsByDigest(folders []string) (map[string]string, error) {
	out := map[string]string{}
	for _, folder := range folders {
		err := filepath.WalkDir(folder, func(p string, d fs.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if !d.Type().IsRegular() || filepath.Ext(p) != ".whl" {
				return nil
			}
			sum, err := fileSHA256(p)
			if err != nil {
				return err
			}
			out[sum] = p
			return nil
		})
		if err != nil {
			return nil, err
		}
	}
	return out, nil
}

func fileSHA256(p string) (string, error) {
	f, err := os.Open(p)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// recordRow is one RECORD line: the wheel path, its SHA-256 and size.
type recordRow struct {
	digest string
	size   int64
}

// The wheel files a RECORD lists without a hash: RECORD itself and its
// signatures.
func unhashedRecordFile(name, distInfo string) bool {
	switch name {
	case distInfo + "/RECORD", distInfo + "/RECORD.jws", distInfo + "/RECORD.p7s":
		return true
	}
	return false
}

// installWheel extracts the wheel at whl into the venv: what it holds
// directly and its .data purelib and platlib go to site, its .data scripts,
// headers and data to the venv's Scripts, Include and root (the wheel
// format's install scheme). Every file must be listed in the wheel's RECORD
// and match its SHA-256 and size there; an entry that is not, a name with a
// drive, a stream or a backslash, and a name that resolves outside its
// destination are refused. Nothing in the wheel runs.
func installWheel(whl, site, venv string) error {
	r, err := zip.OpenReader(whl)
	if err != nil {
		return err
	}
	defer r.Close()
	distInfo := ""
	for _, f := range r.File {
		dir, base := path.Split(f.Name)
		if base == "RECORD" && strings.HasSuffix(dir, ".dist-info/") && strings.Count(dir, "/") == 1 {
			if distInfo != "" {
				return fmt.Errorf("%s holds two RECORDs", whl)
			}
			distInfo = strings.TrimSuffix(dir, "/")
		}
	}
	if distInfo == "" {
		return fmt.Errorf("%s holds no RECORD", whl)
	}
	record, err := readRecord(r, distInfo)
	if err != nil {
		return fmt.Errorf("%s: %w", whl, err)
	}
	data := strings.TrimSuffix(distInfo, ".dist-info") + ".data/"
	checked := map[string]bool{}
	for _, f := range r.File {
		if f.FileInfo().IsDir() {
			continue
		}
		name := f.Name
		if strings.ContainsAny(name, `:\`) || strings.HasPrefix(name, "/") {
			return fmt.Errorf("%s holds the entry %q", whl, name)
		}
		row, listed := record[name]
		if !listed && !unhashedRecordFile(name, distInfo) {
			return fmt.Errorf("%s holds %s, which its RECORD does not list", whl, name)
		}
		root, rel := site, name
		if rest, ok := strings.CutPrefix(name, data); ok {
			scheme, inside, _ := strings.Cut(rest, "/")
			switch scheme {
			case "purelib", "platlib":
				root, rel = site, inside
			case "scripts":
				root, rel = filepath.Join(venv, "Scripts"), inside
			case "headers":
				root, rel = filepath.Join(venv, "Include", strings.TrimSuffix(path.Base(distInfo), ".dist-info")), inside
			case "data":
				root, rel = venv, inside
			default:
				return fmt.Errorf("%s holds %s, in no install scheme", whl, name)
			}
		}
		dest := filepath.Join(root, filepath.FromSlash(rel))
		dir := filepath.Dir(dest)
		if err := sandbox.CheckLocalPlacement(dest); err != nil {
			return fmt.Errorf("%s holds %s, whose destination is unsafe: %w", whl, name, err)
		}
		if !checked[dir] {
			if err := os.MkdirAll(root, 0o700); err != nil {
				return err
			}
			same, _ := identity.SameFile(dir, root)
			inside, err := identity.Within(dir, root)
			if err != nil || !same && !inside {
				return fmt.Errorf("%s holds %s, outside %s", whl, name, root)
			}
			if err := os.MkdirAll(dir, 0o700); err != nil {
				return err
			}
			if err := sandbox.CheckLocalPlacement(dir); err != nil {
				return err
			}
			checked[dir] = true
		}
		if err := sandbox.CheckLocalPlacement(dest); err != nil {
			return err
		}
		if err := extract(f, dest, row, listed && row.digest != ""); err != nil {
			return fmt.Errorf("%s: %s: %w", whl, name, err)
		}
	}
	for name, row := range record {
		if row.digest == "" {
			continue
		}
		if _, err := r.Open(name); err != nil {
			return fmt.Errorf("%s: its RECORD lists %s, which it does not hold", whl, name)
		}
	}
	return nil
}

// readRecord reads the RECORD of distInfo: each listed path with its
// SHA-256 (hex) and size; a path listed without a hash may only be RECORD
// itself or its signature.
func readRecord(r *zip.ReadCloser, distInfo string) (map[string]recordRow, error) {
	f, err := r.Open(distInfo + "/RECORD")
	if err != nil {
		return nil, err
	}
	defer f.Close()
	rows, err := csv.NewReader(f).ReadAll()
	if err != nil {
		return nil, fmt.Errorf("RECORD: %w", err)
	}
	out := map[string]recordRow{}
	for _, row := range rows {
		if len(row) != 3 {
			return nil, fmt.Errorf("RECORD row %q", row)
		}
		name := row[0]
		if row[1] == "" {
			if !unhashedRecordFile(name, distInfo) {
				return nil, fmt.Errorf("RECORD lists %s without a hash", name)
			}
			out[name] = recordRow{}
			continue
		}
		b64, ok := strings.CutPrefix(row[1], "sha256=")
		if !ok {
			return nil, fmt.Errorf("RECORD hashes %s with %q, not sha256", name, row[1])
		}
		sum, err := base64.RawURLEncoding.DecodeString(b64)
		if err != nil || len(sum) != sha256.Size {
			return nil, fmt.Errorf("RECORD hash of %s: %q", name, row[1])
		}
		size, err := strconv.ParseInt(row[2], 10, 64)
		if err != nil || size < 0 {
			return nil, fmt.Errorf("RECORD size of %s: %q", name, row[2])
		}
		out[name] = recordRow{digest: hex.EncodeToString(sum), size: size}
	}
	return out, nil
}

// extract writes one wheel entry to dest, checking its bytes against row
// when check is set; a file that fails the check is removed.
func extract(f *zip.File, dest string, row recordRow, check bool) error {
	src, err := f.Open()
	if err != nil {
		return err
	}
	defer src.Close()
	out, err := os.OpenFile(dest, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0o600)
	if err != nil {
		return err
	}
	h := sha256.New()
	n, err := io.Copy(io.MultiWriter(out, h), src)
	if cerr := out.Close(); err == nil {
		err = cerr
	}
	if err == nil && check && (n != row.size || hex.EncodeToString(h.Sum(nil)) != row.digest) {
		err = errors.New("its bytes are not the ones its RECORD lists")
	}
	if err != nil {
		//unchecked: the extraction failure is the error returned
		os.Remove(dest)
	}
	return err
}
