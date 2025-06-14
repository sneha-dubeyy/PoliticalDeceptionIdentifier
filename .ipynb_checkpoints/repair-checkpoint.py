import os
import shutil
import zipfile
import urllib.request

print("💣 Nuking old punkt and punkt_tab folders...")
nltk_tokenizer_dir = os.path.expanduser('~/nltk_data/tokenizers')
shutil.rmtree(os.path.join(nltk_tokenizer_dir, 'punkt'), ignore_errors=True)
shutil.rmtree(os.path.join(nltk_tokenizer_dir, 'punkt_tab'), ignore_errors=True)

base_nltk = os.path.expanduser('~/nltk_data')
os.makedirs(nltk_tokenizer_dir, exist_ok=True)

# Download the full nltk_data repo
zip_url = "https://github.com/nltk/nltk_data/archive/refs/heads/gh-pages.zip"
local_zip = os.path.join(base_nltk, 'nltk_data-gh-pages.zip')

print("⬇️ Downloading full NLTK punkt repo...")
urllib.request.urlretrieve(zip_url, local_zip)

print("🗃️ Unzipping...")
with zipfile.ZipFile(local_zip, 'r') as zip_ref:
    zip_ref.extractall(base_nltk)

# Try to locate punkt and punkt_tab inside the extracted mess
extracted_base = os.path.join(base_nltk, 'nltk_data-gh-pages')
possible_tokenizers = os.path.join(extracted_base, 'tokenizers')
punkt_src = os.path.join(possible_tokenizers, 'punkt')
punkt_tab_src = os.path.join(possible_tokenizers, 'punkt_tab')

print("🔎 Verifying extracted structure...")
if not os.path.isdir(punkt_src) or not os.path.isdir(punkt_tab_src):
    raise FileNotFoundError("❌ Could not find 'punkt' or 'punkt_tab' folders in extracted zip. Something went wrong.")

print("📦 Installing punkt and punkt_tab folders...")
shutil.copytree(punkt_src, os.path.join(nltk_tokenizer_dir, 'punkt'))
shutil.copytree(punkt_tab_src, os.path.join(nltk_tokenizer_dir, 'punkt_tab'))

# Clean up
print("🧹 Cleaning up...")
shutil.rmtree(extracted_base, ignore_errors=True)
os.remove(local_zip)

print("✅ punkt and punkt_tab installed successfully!")
print("🎯 Now run: python main.py")
