package tools

import "errors"

// errStopWalk is a sentinel used internally to break out of filepath.WalkDir
// once a result limit has been reached; it is never surfaced to callers.
var errStopWalk = errors.New("stop walk: limit reached")
