import Trie from './trie';

function processInput(str) {
  const normalized = str.normalize('NFKC');
  return '▁' + normalized.replace(/ /g, '▁');
}

class Tokenizer {
  constructor(vocab) {
    this.vocab = vocab;
    this.trie = new Trie();

    // The first five tokens are reserved for unk, control symbols, and
    // user-defined symbols.
    for (let i = 6; i < this.vocab.length; i++) {
      this.trie.insert(this.vocab[i][0], this.vocab[i][1], i);
    }
  }

  encode(input) {
    const nodes = [];
    const words = [];
    const best = [];

    input = processInput(input);
    const symbols = [];
    for (const symbol of input) {  // unicode-aware iteration
      symbols.push(symbol);
    }

    for (let i = 0; i <= symbols.length; i++) {
      nodes.push({});
      words.push('');
      best.push(0);
    }

    for (let i = 0; i < symbols.length; i++) {
      const matches = this.trie.commonPrefixSearch(symbols.slice(i));

      for (let j = 0; j < matches.length; j++) {
        const piece = matches[j];
        const obj = {key: piece[0], score: piece[1], index: piece[2]};

        const endPos = [...piece[0]].length;  // unicode aware
        if (typeof nodes[i + endPos][i] === 'undefined') {
          nodes[i + endPos][i] = [];
        }

        nodes[i + endPos][i].push(obj);
      }
    }

    for (let endPos = 0; endPos <= symbols.length; endPos++) {
      for (let startPos in nodes[endPos]) {
        const arr = nodes[endPos][startPos];

        for (let j = 0; j < arr.length; j++) {
          const word = arr[j];
          const score = word.score + best[endPos - word.key.length];

          if (best[endPos] === 0 || score >= best[endPos]) {
            best[endPos] = score;
            words[endPos] = arr[j].index;
          }
        }
      }
    }

    const results = [];

    let iter = words.length - 1;
    while (iter > 0) {
      results.push(words[iter]);
      iter -= this.vocab[words[iter]][0].length;
    }

    // Merge contiguous unks.
    const merged = [];
    let isPreviousUnk = false;
    for (let i = 0; i < results.length; i++) {
      const id = results[i];
      if (!(isPreviousUnk && id === 0)) {
        merged.push(id);
      }

      isPreviousUnk = id === 0;
    }

    return merged.reverse();
  }
}

export default Tokenizer;