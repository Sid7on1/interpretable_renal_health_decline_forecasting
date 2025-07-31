import logging
import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExplanationGenerator:
    """
    Generates explanations for predicted eGFR values using abductive reasoning.

    ...

    Attributes
    ----------
    model : transformers.AutoModelForSeq2SeqLM
        The trained language model for generating explanations.
    tokenizer : transformers.AutoTokenizer
        The tokenizer associated with the trained language model.
    device : torch.device
        The device (CPU or GPU) on which the model is loaded.
    max_length : int
        Maximum length for generated explanations.
    min_length : int
        Minimum length for generated explanations.
    temperature : float
        Temperature parameter for generation. Controls the randomness of token generation.
    top_p : float
        Top-p parameter for generation. Controls the diversity of generated tokens.
    top_k : int
        Top-k parameter for generation. Controls the number of most probable tokens to consider.
    repetition_penalty : float
        Repetition penalty for generation. Discourages repetitive generation.
    num_return_sequences : int
        Number of explanations to generate for each input.
    block_illegal_attributes : bool
        Whether to block generation of explanations containing illegal attributes.
    illegal_attributes : list of str
        List of attributes that are not allowed to be generated.
    velocity_threshold : float
        Velocity threshold above which a decline is considered significant.
    forbidden_tokens : list of str
        Tokens that are forbidden from being generated.

    Methods
    -------
    load_model_and_tokenizer(model_name_or_path, tokenizer_name_or_path)
        Load the language model and tokenizer from pretrained models or local paths.
    generate_explanation(context, predicted_egfr, true_egfr)
        Generate an explanation for the predicted eGFR value based on the input context.
    _filter_illegal_attributes(explanation)
        Filter out illegal attributes from the generated explanation.
    _apply_velocity_threshold(context, predicted_egfr, true_egfr)
        Apply the velocity threshold to determine if a significant decline is present.
    _generate_multiple(inputs, max_outputs=5)
        Generate multiple explanations by sampling multiple times.
    """

    def __init__(self, model_name_or_path, tokenizer_name_or_path, device='cpu', max_length=50, min_length=10, temperature=0.7,
                 top_p=0.9, top_k=0, repetition_penalty=1.0, num_return_sequences=3, block_illegal_attributes=True,
                 illegal_attributes=None, velocity_threshold=0.15, forbidden_tokens=None):
        """
        Initialize the ExplanationGenerator class.

        Parameters
        ----------
        model_name_or_path : str
            Name or path of the trained language model.
        tokenizer_name_or_path : str
            Name or path of the tokenizer associated with the language model.
        device : str, optional
            Device on which to load the model (default is 'cpu').
        max_length : int, optional
            Maximum length for generated explanations (default is 50).
        min_length : int, optional
            Minimum length for generated explanations (default is 10).
        temperature : float, optional
            Temperature parameter for generation (default is 0.7). Controls the randomness of token generation.
        top_p : float, optional
            Top-p parameter for generation (default is 0.9). Controls the probability mass of considered tokens.
        top_k : int, optional
            Top-k parameter for generation (default is 0). Controls the number of highest probability tokens to keep.
        repetition_penalty : float, optional
            Repetition penalty (default is 1.0). Discourages repetitive generation.
        num_return_sequences : int, optional
            Number of explanations to generate for each input (default is 3).
        block_illegal_attributes : bool, optional
            Whether to block generation of explanations containing illegal attributes (default is True).
        illegal_attributes : list of str, optional
            List of attributes that are not allowed to be generated (default is None).
        velocity_threshold : float, optional
            Velocity threshold above which a decline is considered significant (default is 0.15).
        forbidden_tokens : list of str, optional
            Tokens that are forbidden from being generated (default is None).
        """
        self.model = None
        self.tokenizer = None
        self.device = device
        self.max_length = max_length
        self.min_length = min_length
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.num_return_sequences = num_return_sequences
        self.block_illegal_attributes = block_illegal_attributes
        self.illegal_attributes = illegal_attributes if illegal_attributes is not None else []
        self.velocity_threshold = velocity_threshold
        self.forbidden_tokens = forbidden_tokens if forbidden_tokens is not None else []

        self.load_model_and_tokenizer(model_name_or_path, tokenizer_name_or_path)

    def load_model_and_tokenizer(self, model_name_or_path, tokenizer_name_or_path):
        """
        Load the language model and tokenizer.

        Parameters
        ----------
        model_name_or_path : str
            Name or path of the trained language model.
        tokenizer_name_or_path : str
            Name or path of the tokenizer associated with the language model.
        """
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

        self.model.to(self.device)

    def generate_explanation(self, context, predicted_egfr, true_egfr):
        """
        Generate an explanation for the predicted eGFR value based on the input context.

        Parameters
        ----------
        context : str
            The input context containing relevant information for explanation generation.
        predicted_egfr : float
            The predicted eGFR value for which an explanation is sought.
        true_egfr : float
            The true eGFR value.

        Returns
        -------
        explanation : str
            The generated explanation for the predicted eGFR value.
        """
        try:
            if predicted_egfr is None or true_egfr is None:
                raise ValueError("Predicted and true eGFR values cannot be None.")

            # Apply velocity threshold to determine if a significant decline is present
            significant_decline = self._apply_velocity_threshold(context, predicted_egfr, true_egfr)

            # Input format: "Explain the predicted eGFR of {predicted_egfr} based on the context: {context}."
            input_text = f"Explain the predicted eGFR of {predicted_egfr} based on the context: {context}."

            # Tokenize input text
            inputs = self.tokenizer.encode(input_text, return_tensors='pt').to(self.device)

            # Generate explanations
            explanations = self._generate_multiple(inputs)

            # Filter illegal attributes from explanations
            if self.block_illegal_attributes:
                explanations = [self._filter_illegal_attributes(explanation) for explanation in explanations]

            # Return the most probable explanation
            explanation = explanations[0]

            # Log generated explanation
            logger.info(f"Generated explanation: {explanation}")

            return explanation

        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            return None

    def _filter_illegal_attributes(self, explanation):
        """
        Filter out illegal attributes from the generated explanation.

        Parameters
        ----------
        explanation : str
            The generated explanation containing potential illegal attributes.

        Returns
        -------
        filtered_explanation : str
            The explanation with illegal attributes removed.
        """
        # Tokenize the generated explanation
        tokens = self.tokenizer.tokenize(explanation)

        # Remove illegal attributes from the token list
        filtered_tokens = [token for token in tokens if token not in self.illegal_attributes]

        # Reconstruct the filtered explanation
        filtered_explanation = ' '.join(filtered_tokens)

        return filtered_explanation

    def _apply_velocity_threshold(self, context, predicted_egfr, true_egfr):
        """
        Apply the velocity threshold to determine if a significant decline is present.

        Parameters
        ----------
        context : str
            The input context containing relevant information.
        predicted_egfr : float
            The predicted eGFR value.
        true_egfr : float
            The true eGFR value.

        Returns
        -------
        significant_decline : bool
            True if a significant decline is detected, False otherwise.
        """
        try:
            if predicted_egfr >= true_egfr:
                return False

            # Extract necessary information from the context
            required_info = "Extract patient information from the context."
            patient_info = self._extract_info(context, required_info)

            # Apply velocity threshold formula
            velocity = (true_egfr - predicted_egfr) / patient_info['time_interval']
            significant_decline = velocity > self.velocity_threshold

            return significant_decline

        except Exception as e:
            logger.error(f"Error applying velocity threshold: {e}")
            return False

    def _generate_multiple(self, inputs, max_outputs=5):
        """
        Generate multiple explanations by sampling multiple times.

        Parameters
        ----------
        inputs : torch.Tensor
            Tokenized input sequences.
        max_outputs : int, optional
            Maximum number of outputs to generate (default is 5).

        Returns
        -------
        explanations : list of str
            List of generated explanations.
        """
        try:
            # Generate explanations multiple times
            outputs = self.model.generate(
                inputs,
                max_length=self.max_length,
                min_length=self.min_length,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                repetition_penalty=self.repetition_penalty,
                do_sample=True,
                num_return_sequences=self.num_return_sequences,
                num_beams=self.num_return_sequences,
                num_beam_groups=self.num_return_sequences,
                diversity_penalty=0.5,
                max_outputs=max_outputs,
                forbidden_tokens=self.forbidden_tokens,
                output_scores=True
            )

            # Decode outputs and get the explanations
            explanations = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

            return explanations

        except Exception as e:
            logger.error(f"Error generating multiple explanations: {e}")
            return []

    def _extract_info(self, context, required_info):
        """
        Extract necessary information from the input context.

        Parameters
        ----------
        context : str
            The input context containing relevant information.
        required_info : str
            The specific information that is required to be extracted.

        Returns
        -------
        extracted_info : dict
            Dictionary containing the extracted information.
        """
        # TODO: Implement the logic to extract necessary information from the context
        # Return a dictionary with the required information
        # Placeholder implementation
        extracted_info = {'time_interval': 30}

        return extracted_info

# Example usage
if __name__ == '__main__':
    model_name = "gpt2"
    tokenizer_name = "gpt2"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize the ExplanationGenerator
    explainer = ExplanationGenerator(model_name, tokenizer_name, device=device)

    # Example context, predicted_egfr, and true_egfr values
    context = "The patient is a 45-year-old male with a history of diabetes and hypertension."
    predicted_egfr = 60.5
    true_egfr = 75.3

    # Generate an explanation
    explanation = explainer.generate_explanation(context, predicted_egfr, true_egfr)
    print(f"Generated explanation: {explanation}")