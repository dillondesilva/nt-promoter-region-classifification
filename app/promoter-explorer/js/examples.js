/** Load examples exported from scripts/generate_promoter_examples.py */
export async function loadExampleCatalog() {
  const res = await fetch('data/examples.json');
  if (!res.ok) {
    throw new Error(`Failed to load examples (${res.status})`);
  }
  return res.json();
}
