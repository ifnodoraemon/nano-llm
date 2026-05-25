import os
import re
import urllib.request
import logging
import argparse
from typing import List

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ==============================================================================
# Pure Python HTML Stripping & Extraction (from scratch)
# ==============================================================================

def strip_html_tags(html_content: str) -> str:
    """
    Strips boilerplate HTML elements (scripts, styles, headers, comments, tags) 
    using regex to extract clean body text from scratch.
    """
    # 1. Remove script and style elements completely
    text = re.sub(r'<(script|style|iframe|header|footer|nav)[^>]*>.*?</\1>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
    
    # 2. Remove HTML comments
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    
    # 3. Remove inline HTML tags (like <p>, <div>, <span>)
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # 4. Replace standard HTML entities
    text = text.replace("&nbsp;", " ")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    text = text.replace("&amp;", "&")
    text = text.replace("&quot;", '"')
    
    # 5. Clean up redundant whitespaces and newlines
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    
    return text.strip()


def crawl_url(url: str) -> str:
    """
    Fetches raw HTML and extracts clean text using urllib (python standard library).
    """
    logger.info(f"Crawling URL: {url}...")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as response:
            html = response.read().decode("utf-8", errors="ignore")
            
        clean_text = strip_html_tags(html)
        logger.info(f"Crawl succeeded! Extracted {len(clean_text)} characters.")
        return clean_text
    except Exception as e:
        logger.error(f"Failed to crawl {url}: {e}")
        return ""

# ==============================================================================
# Resilient Corpus Generator (Offline Fallback)
# ==============================================================================

def generate_local_corpus(output_dir: str):
    """
    Generates a high-quality local text corpus on deep learning, programming history, 
    and physics to guarantee the pipeline remains 100% executable offline.
    """
    logger.info("Generating high-quality local text corpus (Offline Fallback)...")
    
    documents = [
        ("deep_learning_history.txt", 
         "Deep learning is a subset of machine learning that is based on artificial neural networks with representation learning. "
         "The history of deep learning began with the perceptron in 1957, invented by Frank Rosenblatt. Rosenblatt's perceptron "
         "was a single-layer neural network designed for image recognition. However, in 1969, Marvin Minsky and Seymour Papert "
         "published the book 'Perceptrons', proving that single-layer networks could not solve non-linear problems like XOR. "
         "This led to a period called the first AI winter. In the 1980s, the backpropagation algorithm was popularized by "
         "Geoffrey Hinton, Yann LeCun, and Yoshua Bengio, allowing deep multi-layer neural networks to be trained efficiently. "
         "The massive breakthrough came in 2012 with AlexNet, a deep convolutional neural network that won the ImageNet challenge "
         "by a massive margin, initiating the modern era of deep learning and generative artificial intelligence."),
         
        ("programming_languages.txt",
         "Programming languages are the primary tools used by developers to communicate instructions to computers. "
         "In the early days of computing, programmers wrote instructions in raw binary machine code. In 1957, John Backus and IBM "
         "developed FORTRAN, the first widely used high-level programming language, which allowed mathematical formulas to be expressed "
         "in readable text. In 1972, Dennis Ritchie at Bell Labs invented the C programming language, which combined the power of low-level "
         "assembly languages with high-level readability. C became the foundation of modern operating systems, including Unix and Windows. "
         "In 1991, Guido van Rossum released Python, designed with a focus on code readability and clean syntax. Today, Python is the "
         "dominant language for machine learning, artificial intelligence, and data science."),
         
        ("quantum_physics.txt",
         "Quantum mechanics is a fundamental theory in physics that describes the physical properties of nature at the scale of atoms and subatomic particles. "
         "In classical physics, energy is continuous, but in 1900, Max Planck proposed that energy is emitted in discrete packets called 'quanta'. "
         "In 1905, Albert Einstein explained the photoelectric effect by proposing that light behaves as packets of energy called photons, "
         "introducing wave-particle duality. In the 1920s, Werner Heisenberg formulated the uncertainty principle, stating that it is impossible "
         "to simultaneously measure both the position and momentum of a particle with absolute precision. Erwin Schrodinger formulated "
         "his wave equation, representing the state of a quantum system as a probability wave function. Quantum mechanics forms the basis of "
         "semiconductor technology, lasers, and modern quantum computing."),
         
        # Duplicate document to test our MinHash deduplication script downstream!
        ("deep_learning_history_duplicate.txt",
         "Deep learning is a subset of machine learning based on artificial neural networks with representation learning. "
         "The history of deep learning began with Rosenblatt's perceptron in 1957. Marvin Minsky and Seymour Papert proved "
         "that single-layer networks could not solve non-linear problems like XOR in 1969. In the 1980s, the backpropagation "
         "algorithm was popularized by Hinton, LeCun, and Bengio, allowing deep multi-layer networks to be trained. "
         "In 2012, AlexNet won the ImageNet challenge by a huge margin, starting the modern deep learning era.")
    ]
    
    for filename, text in documents:
        file_path = os.path.join(output_dir, filename)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)
        logger.info(f"Exported local corpus file: {file_path}")

# ==============================================================================
# Main Runner
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="nano-llm: zero-dependency raw Web Crawler & Stripper")
    parser.add_argument("--output_dir", type=str, default="./data/raw_crawled", help="Output directory to save text files")
    parser.add_argument("--source", type=str, choices=["crawl", "local"], default="local", help="Source type: crawl online URLs or generate local fallback corpus")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.source == "local":
        generate_local_corpus(args.output_dir)
    else:
        # Default list of educational URLs (Wikipedia articles on AI, physics, programming history)
        urls = [
            "https://en.wikipedia.org/wiki/Deep_learning",
            "https://en.wikipedia.org/wiki/Programming_language",
            "https://en.wikipedia.org/wiki/Quantum_mechanics"
        ]
        
        crawled_any = False
        for idx, url in enumerate(urls):
            text = crawl_url(url)
            if text:
                file_path = os.path.join(args.output_dir, f"wiki_article_{idx+1}.txt")
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(text)
                logger.info(f"Saved crawled article to: {file_path}")
                crawled_any = True
                
        # If online crawls fail due to connection issues on local docker/environment, fallback
        if not crawled_any:
            logger.warning("All online crawls failed. Falling back to local offline corpus generation.")
            generate_local_corpus(args.output_dir)
            
    logger.info(f"✅ Web Crawling & HTML Stripping finished successfully. Output directory: {args.output_dir}/")

if __name__ == "__main__":
    main()
