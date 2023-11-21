import re
import sys
from collections import defaultdict
from multiprocessing import Pool, cpu_count

def preprocess_text(text):
  text = text.lower()
  text = re.sub(r'[^a-záéíóúâêîôûãõàèìòùäëïöüçñ\s]', '', text)
  return text


def count_words(text):
  word_count = defaultdict(int)
  words = text.split()
  for word in words:
    if len(word) >= 2:
      word_count[word] += 1
  return word_count


def freq_n_words(word_count, n):
  sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
  return sorted_words[:n]


def freq_word(word_count, word):
  return word_count.get(word, 0)


def calculate_tfidf(docs, term):
  tfidf_scores = defaultdict(float)
  term_freq = defaultdict(int)
  doc_freq = defaultdict(int)

  for i, doc in enumerate(docs):
    doc_text = preprocess_text(doc)
    words = set(doc_text.split())

    if term in words:
        term_freq[f"Documento {i + 1}"] = doc_text.count(term)

        for word in words:
          doc_freq[word] += 1

    for doc, freq in term_freq.items():
      tf = freq / len(docs)
      idf = 1 / doc_freq[term]
      tfidf_scores[doc] = tf * idf

    sorted_docs = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_docs


def process_chunk(chunk):
  processed_text = preprocess_text(chunk)
  return count_words(processed_text)


def read_large_file(file_path, chunk_size=1024 * 1024):
  with open(file_path, 'rb') as file:
    while True:
      chunk = file.read(chunk_size)
      if not chunk:
        break
      yield chunk.decode('utf-8', errors='ignore')


def read_large_file_lines(file_path):
  with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
    for line in file:
      yield line


def count_term_occurrences(docs, term):
  term_counts = defaultdict(int)

  for i, doc in enumerate(docs):
    lines = doc.split('\n')  
    for line_number, line in enumerate(lines, start=1):
      doc_text = preprocess_text(line)
      term_count = doc_text.split().count(term)
      term_counts[f"Documento {i + 1}, Linha {line_number}"] = term_count

    total_occurrences = sum(term_counts.values())
    term_counts["Total"] = total_occurrences

    return term_counts


def main():
    if len(sys.argv) < 2:
      print("Uso: python indexer.py <opção> <argumentos>")
      sys.exit(1)

    option = sys.argv[1]

    if option == "--freq":
      if len(sys.argv) != 4:
        print("Uso: python indexer.py --freq N ARQUIVO")
        sys.exit(1)
      n = int(sys.argv[2])
      filename = sys.argv[3]

      pool = Pool(processes=cpu_count())
      word_count = defaultdict(int)

      for chunk_word_count in pool.imap_unordered(process_chunk, read_large_file(filename)):
        for word, count in chunk_word_count.items():
          word_count[word] += count

      pool.close()
      pool.join()

      result = freq_n_words(word_count, n)
      print(result)

    elif option == "--freq-word":
      if len(sys.argv) != 4:
        print("Uso: python indexer.py --freq-word PALAVRA ARQUIVO")
        sys.exit(1)
      word = sys.argv[2]
      filename = sys.argv[3]

      pool = Pool(processes=cpu_count())
      word_count = defaultdict(int)

      for chunk_word_count in pool.imap_unordered(process_chunk, read_large_file(filename)):
        for chunk_word, count in chunk_word_count.items():
          if chunk_word == word:
            word_count[word] += count

      pool.close()
      pool.join()

      result = freq_word(word_count, word)
      print(result)

    elif option == "--search":
      if len(sys.argv) < 4:
        print("Uso: python indexer.py --search TERMO ARQUIVO [ARQUIVO ...]")
        sys.exit(1)
      term = sys.argv[2]
      filenames = sys.argv[3:]

      results = defaultdict(int)

      for filename in filenames:
        docs = []
        with open(filename, 'r', encoding='utf-8', errors='ignore') as file:
          docs.append(file.read())

        result = count_term_occurrences(docs, term)
        results[filename] = result["Total"]

        total_occurrences = sum(results.values())
        print(f"{', '.join(filenames)}: {total_occurrences} ocorrência(s) do termo '{term}'")

        for filename, count in results.items():
          print(f"{filename}: {count} ocorrência(s) do termo '{term}'")

    else:
      print("Opção não reconhecida.")
      sys.exit(1)

if __name__ == "__main__":
  main()
