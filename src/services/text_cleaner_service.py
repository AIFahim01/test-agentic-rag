import re
# import contractions
from string import punctuation


class TextCleaner:
    def lower_text(self, text):
        """
        Converts all characters in the text to lowercase.

        @param text: The original text.
        @return: Lowercased text.

        Example:
            lower_text("Hello WORLD") -> "hello world"
        """
        return text.lower()

    def remove_numbers(self, text):
        """
        Removes all digits from the text.

        @param text: The original text containing numbers.
        @return: Text with numbers removed.

        Example:
            remove_numbers("Year 2020") -> "Year "
        """
        return re.sub(r'\d+', '', text)

    def remove_punctuation(self, text):
        """
        Removes all punctuation from the text.

        @param text: The original text containing punctuation.
        @return: Text with punctuation removed.

        Example:
            remove_punctuation("Hello, world!") -> "Hello world"
        """
        clean_text = re.sub(f"[{re.escape(punctuation)}]", "", text)
        return clean_text

    def remove_html_tags(self, text):
        """
        Removes HTML tags from the text.

        @param text: The original text containing HTML tags.
        @return: Text with HTML tags removed.

        Example:
            remove_html_tags("<p>Hello World!</p>") -> "Hello World!"
        """
        clean_text = re.sub(r"<.*?>", "", text)
        return clean_text

    def remove_whitespace(self, text):
        """
        Removes excessive whitespace from the text, including tabs and newlines.

        @param text: The original text with possible excessive whitespace.
        @return: Text with extra whitespace removed.

        Example:
            remove_whitespace("  Hello   World  ") -> "Hello World"
        """
        clean_text = " ".join(text.split())
        return clean_text


    def fix_encoding(self, text):
        """
        Attempts to fix encoding issues in the text.

        @param text: The original text with potential encoding issues.
        @return: Text with encoding fixed, or 'Encoding Error' if unsuccessful.

        Example:
            fix_encoding("This is an encoded string") -> "This is an encoded string"
        """
        try:
            decoded_text = text.encode('utf-8').decode('utf-8')
        except UnicodeDecodeError:
            decoded_text = 'Encoding Error'
        return decoded_text

    def remove_url(self, text):
        """
        Removes URLs from the text.

        @param text: The original text containing URLs.
        @return: Text with URLs removed.

        Example:
            remove_url("Check this https://example.com") -> "Check this "
        """
        url_pattern = (r"\b(?:https?://|www\.)\S+\b|(?:(?<![@\w])\b\w+\.("
                       r"com|net|org|info|coop|int|co\.uk|org\.uk|ac\.uk|uk)\b)")
        return re.sub(url_pattern, '', text)

    def remove_emojis(self, text):
        """
        Removes emojis from the text.

        @param text: The original text containing emojis.
        @return: Text with emojis removed.

        Example:
            remove_emojis("Hello 😊") -> "Hello "
        """
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)

    def fix_contractions(self, text):
        """
        Expands contractions in the text.

        @param text: The original text containing contractions.
        @return: Text with contractions expanded.

        Example:
            fix_contractions("I'm happy") -> "I am happy"
        """
        return contractions.fix(text)

    def clean_text(self, text):
        """
        Applies a sequence of cleaning processes to the text.

        @param text: The original text to be cleaned.
        @return: Fully cleaned text.

        Example:
            clean_text("Hello, I'm Anwar! Visit my blog at http://anwar.com") -> "hello anwar visit my blog"
        """
        text = self.fix_encoding(text)
        text = self.remove_url(text)
        text = self.lower_text(text)
        text = self.remove_html_tags(text)
        text = self.remove_emojis(text)
        text = self.remove_numbers(text)
        text = self.remove_punctuation(text)
        text = self.remove_whitespace(text)
        # text = self.fix_contractions(text)
        return text