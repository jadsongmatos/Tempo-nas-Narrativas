from transformers import AutoTokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import duckdb
from tqdm import tqdm
from datasets import load_dataset

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B", legacy=False)

# Define the length function
def length_function(text: str) -> int:
    return len(tokenizer(text)['input_ids'])

# Define the splitter_length function
def splitter_length(text: str, name: str, use_tqdm: bool = True):
    sizes_length = [16, 32, 64, 128, 256, 512, 1024]
    
    for size in sizes_length:
        overlap = int(size / 5)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=size, 
            chunk_overlap=overlap,
            length_function=length_function,
        )

        splits = splitter.split_text(text)
        splits = [split for split in splits if len(split) > 1]  # Filter splits with length > 1

        if use_tqdm:
            iterator = tqdm(splits, desc=f"Processing chunks of size {size}")
        else:
            iterator = splits

        inserts = []
        last_found = 0  # Track current position in the text

        for split in iterator:
            # Adjust search position to include overlap
            search_start = max(last_found - overlap, 0)
            indice = text.find(split, search_start)
            
            if indice == -1:
                # If not found from search_start, search globally
                indice = text.find(split)
                if indice == -1:
                    print(f"Split not found: {split[:30]}...")
                    continue

            inserts.append((split, indice, name))
            # Update current position considering the overlap
            last_found = indice + len(split) - overlap

        # Batch insert data into the database
        if inserts:
            try:
                conn.executemany("""
                    INSERT INTO dataset (content, indice, name)
                    VALUES (?, ?, ?)
                """, inserts)
            except Exception as e:
                print(f"Error inserting data into DuckDB: {e}")

# Connect or create the DuckDB database
conn = duckdb.connect('/app/duckdb/fineweb.duckdb')

# Create the dataset table if it doesn't exist
conn.execute("""
CREATE SEQUENCE IF NOT EXISTS serial;
             
CREATE TABLE IF NOT EXISTS dataset (
    id INTEGER DEFAULT nextval('serial'),
    name VARCHAR(255),
    content TEXT,
    indice INTEGER
)
""")

# Load the dataset in streaming mode
fw = load_dataset("/app/fineweb/", streaming=True)

# Check available splits (e.g., 'train', 'test', etc.)
print(fw)

# Process each sample in the 'train' split
for sample in tqdm(fw['train'], desc="Processing samples"):
    splitter_length(sample['text'], sample['url'], use_tqdm=False)

# Close the database connection
conn.close()

"""
sudo docker run --rm \
  -v "$(pwd)/fineweb:/app/fineweb" \ 
  -v "$(pwd)/duckdb:/app/duckdb" \
  my_script_image:latest
"""
