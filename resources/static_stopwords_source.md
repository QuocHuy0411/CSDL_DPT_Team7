# C2 static stoplist

Source: https://github.com/stopwords/vietnamese-stopwords

Pinned revision: `a453d389e1b52e20748ca83ddd8b0faebb04f5aa`.
File: `vietnamese-stopwords.txt`, 1,942 nonblank source entries (counted on download).
Local snapshot: `static_stopwords.txt`. Runtime normalizes NFC/case and joins phrase syllables with underscores to match M1 token convention. Does not split a compound stopword into independent syllables. With the development syllable tokenizer, multi-syllable entries cannot match; effective matched vocabulary count is recorded per index. C2 applies the static source as supplied, including its negations; M1 semantic guards apply to C3/C4 and are not silently imposed on C2.

MIT License

Copyright (c) 2015 Van-Duyet Le

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
