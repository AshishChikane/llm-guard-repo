# Anonymize Scanner

The `Anonymize` Scanner acts as your digital guardian, ensuring your user prompts remain confidential and free from
sensitive data exposure.

## What is PII?

PII, an acronym for Personally Identifiable Information, is the cornerstone of an individual's digital identity. Leaks
or mishandling of PII can unleash a storm of problems, from privacy breaches to identity theft. Global regulations,
including GDPR and HIPAA, underscore the significance of PII by laying out strict measures for its protection.
Furthermore, any unintentional dispatch of PII to LLMs can proliferate this data across various storage points, thus
raising the stakes.

## Attack

Sometimes, Language Learning Models (or LLMs) can accidentally share private info from the prompts they get. This can be
bad because it might let others see or use this info in the wrong way.

To stop this from happening, we use the `Anonymize` scanner. It makes sure user prompts don’t have any private details
before the model sees them.

## PII Entities

- **Credit Cards**: Formats mentioned in [Wikipedia](https://en.wikipedia.org/wiki/Payment_card_number).
  - `4111111111111111`
  - `378282246310005` (American Express)
  - `30569309025904` (Diners Club)
- **Person**: A full person name, which can include first names, middle names or initials, and last names.
  - `John Doe`
- **PHONE_NUMBER**:
  - `5555551234`
- **URL**: A URL (Uniform Resource Locator), unique identifier used to locate a resource on the Internet.
  - `https://laiyer.ai`
- **E-mail Addresses**: Standard email formats.
  - `john.doe@laiyer.ai`
  - `john.doe[AT]laiyer[DOT]ai`
  - `john.doe[AT]laiyer.ai`
  - `john.doe@laiyer[DOT]ai`
- **IPs**: An Internet Protocol (IP) address (either IPv4 or IPv6).
  - `192.168.1.1` (IPv4)
  - `2001:db8:3333:4444:5555:6666:7777:8888` (IPv6)
- **UUID**:
  - `550e8400-e29b-41d4-a716-446655440000`
- **US Social Security Number (SSN)**:
  - `111-22-3333`
- **Crypto wallet number**: Currently only Bitcoin address is supported.
  - `1Lbcfr7sAHTD9CgdQo3HTMTkV8LK4ZnX71`
- **IBAN Code**: The International Bank Account Number (IBAN) is an internationally agreed system of identifying bank
  accounts across national borders to facilitate the communication and processing of cross border transactions with a
  reduced risk of transcription errors.
  - `DE89370400440532013000`

## Features

- **Integration with [Presidio Analyzer](https://github.com/microsoft/presidio/)**: Leverages the Presidio Analyzer
  library, crafted with spaCy, flair and transformers libraries, for precise detection of private data.
- **Enhanced Detection**: Beyond Presidio Analyzer's capabilities, the scanner recognizes specific patterns like Email,
  US SSN, UUID, and more.
- **Entities Support**:
  - Peek at
    our [default entities](https://github.com/laiyer-ai/llm-guard/blob/main/llm_guard/input_scanners/anonymize.py#L26-L40).
  - View
    the [Presidio's supported entities](https://microsoft.github.io/presidio/supported_entities/#list-of-supported-entities).
  - And, we've
    got [custom regex patterns](https://github.com/laiyer-ai/llm-guard/blob/main/llm_guard/resources/sensisitive_patterns.json)
    too!
- **Tailored Recognizers**:
  - Balance speed vs. accuracy of the recognizers.
  - **Top Pick: [dslim/bert-base-NER](https://huggingface.co/dslim/bert-base-NER)**
  - Alternatives: [dslim/bert-large-NER](https://huggingface.co/dslim/bert-large-NER).

!!! info

    Current entity detection functionality is English-specific.

## Get Started

Initialize the `Vault`: The Vault archives data that's been redacted.

```python
from llm_guard.vault import Vault

vault = Vault()
```

Configure the `Anonymize` Scanner:

```python
from llm_guard.input_scanners import Anonymize
from llm_guard.input_scanners.anonymize_helpers import BERT_LARGE_NER_CONF

scanner = Anonymize(vault, preamble="Insert before prompt", allowed_names=["John Doe"], hidden_names=["Test LLC"],
                    recognizer_conf=BERT_LARGE_NER_CONF)
sanitized_prompt, is_valid, risk_score = scanner.scan(prompt)
```

- `preamble`: Directs the LLM to bypass specific content.
- `hidden_names`: Transforms specified names to formats like `[REDACTED_CUSTOM_1]`.
- `entity_types`: Opt for particular information types to redact.
- `regex_pattern_groups_path`: Input a path for personalized patterns.
- `use_faker`: Substitutes eligible entities with fabricated data.
- `recognizer_conf`: Configures recognizer for the PII data detection.
- `threshold`: Sets the acceptance threshold (Default: `0`).

Retrieving Original Data: To revert to the initial data, utilize the [Deanonymize](../output_scanners/deanonymize.md)
scanner.

## Optimizations

### ONNX

The scanner can be optimized by converting the model to ONNX format. This can be done by setting the `use_onnx`.

Make sure to install the `onnxruntime` package:

```sh
pip install llm-guard[onnxruntime]
```

## Benchmarks

Environment:

- Platform: Amazon Linux 2
- Python Version: 3.11.6

Run the following script:

```sh
python benchmarks/run.py input Anonymize
```

Results:

| Instance                | Input Length | Test Times | Latency Variance | Latency 90 Percentile | Latency 95 Percentile | Latency 99 Percentile | Average Latency (ms) | QPS      |
|-------------------------|--------------|------------|------------------|-----------------------|-----------------------|-----------------------|----------------------|----------|
| AWS m5.xlarge           | 317          | 5          | 6.11             | 255.64                | 294.57                | 325.71                | 177.13               | 1789.64  |
| AWS m5.xlarge with ONNX | 317          | 5          | 0.73             | 155.64                | 169.13                | 179.93                | 128.64               | 2464.29  |
| AWS g5.xlarge GPU       | 317          | 5          | 38.50            | 321.59                | 419.60                | 498.01                | 125.18               | 2532.35  |
