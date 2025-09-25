## Usage

Prepare python environment

```bash
pip install -r requirements.txt
```

For java, you need to install JDK17 and maven.

```bash
wget https://download.oracle.com/java/17/archive/jdk-17.0.12_linux-x64_bin.tar.gz
wget https://dlcdn.apache.org/maven/maven-3/3.9.11/binaries/apache-maven-3.9.11-bin.tar.gz
tar -zxvf jdk-17.0.12_linux-x64_bin.tar.gz
tar -zxvf maven-mvnd-1.0.2-linux-amd64.tar.gz
export JAVA_HOME=~/jdk-17.0.12
export PATH=$PATH:$JAVA_HOME/bin
export MAVEN_HOME=~/apache-maven-3.9.11
export PATH=$PATH:$MAVEN_HOME/bin
```

For cpp, you need to support C++20 standard.

After environment preparation, you can run the following command to evaluate the model.

```bash
python testeval.py --n 5 --model_path /ckpts/Dream-v0-Instruct-7B -l python -t 1.0 --max_length 128 --accelerate
```

## Results

All the meta results of the experiments we mentioned in the paper are available in the `results` folder. And some figures in our paper are also available in the `results` folder.

By running `images.ipynb`, you can reproduce the figures in the paper.

## Reference

[DiffuCoder model](https://huggingface.co/apple/DiffuCoder-7B-cpGRPO/tree/main)

[Dream-Instruct model](https://huggingface.co/Dream-org/Dream-v0-Instruct-7B/tree/main)

[TestEval benchmark](https://github.com/LLM4SoftwareTesting/TestEval)

**IMPORTANT**: The evaluation code will replace `generate_utils.py` with `./generate_utils_diffucoder.py` to enable acceleration.

