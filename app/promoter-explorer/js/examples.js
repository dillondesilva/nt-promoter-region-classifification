const SEQ_LEN = 300;

function padSeq(s) {
  const clean = s.toUpperCase().replace(/[^ATGC]/g, '');
  if (clean.length >= SEQ_LEN) return clean.slice(0, SEQ_LEN);
  const fill = 'ATGC';
  let out = clean;
  while (out.length < SEQ_LEN) out += fill[out.length % 4];
  return out;
}

export const EXAMPLES = [
  {
    title: 'TATA-box promoter (HBB)',
    desc: 'Human β-globin — classic TATA-containing promoter region',
    seq: padSeq(
      'GGGCATAGAAAGTCAGGGCAGAGCCATCTATTGCTTACATTTGCTTCTGACACAACTGTGTTCACTAGCAACCTCAAACAGACACCATGGTGCATCTGACTCCTGAGGAGAAGTCTGCCGTTACTGCCCTGTGGGGCAAGGTG'
    ),
  },
  {
    title: 'CpG island promoter (GAPDH)',
    desc: 'Housekeeping gene — GC-rich CpG island promoter',
    seq: padSeq(
      'CCCCGCCCGCCGCGCCCGGTCCCATCCCGGCCCCCGGCCCGCGCCCCGGCCCGCCCCGCGCCCCGGCCCGTTCCCAGCCCCGCCTCCCGAGCCCCGCGCCCCGAATCCCGCGGCCGCGCCCCCTCCCCCACCCCCAGGTTCC'
    ),
  },
  {
    title: 'Non-promoter (TP53 intron)',
    desc: 'Intronic region — no promoter activity expected',
    seq: padSeq(
      'ACTGAATCTAGATGTCATCTGGAGCAGCTGGTGATGGGTAATGCTGACTCAGCCTTGTGGAATCAGATTCCAATCTTGGCTCACTGAGATGTTACTGACATTTCACTTCCTGATTGATGGTGATGTCAAACCTCTTACTGGAA'
    ),
  },
  {
    title: 'Bidirectional promoter (BRCA1)',
    desc: 'Shared promoter between BRCA1 and NBR2',
    seq: padSeq(
      'CGCGCGATTCGCGCGCCCGGCTCCGCCCGCCGCCCCGGCCCGCGCGTTCTTCCCGCCTCCCGGACCCCGCGCCCCGCCCGCGCTCCGCCCGCCCCTTCCTTTCCGCGGCCCCGCCCGCCTTCCCGCGCCTCCCTTCCCGCCC'
    ),
  },
];
