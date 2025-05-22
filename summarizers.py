from transformers import pipeline
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    logging.info("Summarization model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading summarization model: {e}")
    summarizer = None

def generate_structured_summary(text, summary_percentage=0.3, structured_data=None):
    if summarizer is None:
        logging.error("Summarization model not loaded, cannot generate summary.")
        return "Summarization model not loaded."

    try:
        input_length = len(text.split())
        max_length = min(int(input_length * summary_percentage), input_length - 1)
        max_length = max(10, max_length)
        min_length = max(5, int(max_length * 0.5))

        logging.info(f"Input length: {input_length}, max_length: {max_length}, min_length: {min_length}")

        if structured_data:
            structured_summary_parts = []
            for item in structured_data:
                if item['type'] == 'heading':
                    structured_summary_parts.append(f"**{item['content'].strip()}**")
                elif item['type'] == 'paragraph':
                    try:
                        # Summarize only if the paragraph is long enough
                        para_length = len(item['content'].split())
                        if para_length < 10:
                            structured_summary_parts.append(item['content'])
                            continue
                        para_max_length = min(int(para_length * summary_percentage), para_length - 1)
                        para_max_length = max(10, para_max_length)
                        para_min_length = max(5, int(para_max_length * 0.5))
                        summary_output = summarizer(
                            item['content'],
                            max_length=para_max_length,
                            min_length=para_min_length,
                            do_sample=False
                        )
                        logging.info(f"Summary generated for paragraph: {summary_output[0]['summary_text']}")
                        structured_summary_parts.append(summary_output[0]['summary_text'])
                    except Exception as e:
                        logging.warning(f"Error summarizing paragraph: {e}")
                        structured_summary_parts.append(item['content'][:200] + "...")
            return "\n\n".join(structured_summary_parts)

        else:
            logging.info(f"Input text: {text[:100]}...")
            summary_output = summarizer(
                text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False
            )
            logging.info(f"Summary output: {summary_output[0]['summary_text']}")
            return summary_output[0]['summary_text']

    except Exception as e:
        logging.error(f"Error during summarization: {e}")
        raise