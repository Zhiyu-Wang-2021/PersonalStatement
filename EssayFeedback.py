from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import CommaSeparatedListOutputParser
import pandas as pd
from langchain.chains import LLMChain
from tqdm import tqdm
from langchain.embeddings import OpenAIEmbeddings


class EssayFeedback:

    def get_advice(self, article, program):
        load_dotenv(find_dotenv())
        llm = ChatOpenAI(
            openai_api_key="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",
            temperature=0,
            model="gpt-4-0613",
            verbose=True,
        )
        prompt = PromptTemplate.from_template("""
        ### Your role
        You are an admission officer from Harvard University. You are reading an application essay from a student who is applying for the {program} program. The student is writing about their background and why they want to study at Harvard. Please write a response to the student to tell how to improve the essay.
        
        ### Requirement of your response
        Your response should be written in a formal and professional tone. You should write 3 points to tell the student how to improve the essay. Each point should start with the line number of the sentence inn the essay, then the original sentence from the essay enclosed in quotation marks, and finally your advice. Separate each points with ||
        
        ### Example of your response
        1|"I am an experienced and passionate public health advocate who has been driven to pursue a graduate degree in the field for several years."|"You should remove this sentence because it is too long and wordy. You should write a shorter sentence to express the same idea."||4|"Dedicated to promoting health justice, equity, and well-being on both local and global levels, I’ve collaborated with multiple non-profit organizations over the years."|"You should include more details about the non-profit organizations you have collaborated with. You should also include more details about the health justice, equity, and well-being you have promoted."||10|"Studying at Harvard would provide me with the tools needed to drive impactful progress while developing my skills as a leader in public health."|"You should focus more on the specific tools you want to learn from Harvard. You should also focus more on the specific skills you want to develop from Harvard."
        
        ### The student's essay
        {article}
        
        ### Your response
        """)
        prompt.format_prompt(article=article, program=program)
        chain = LLMChain(llm=llm, prompt=prompt)
        result = chain.run(article=article, program=program)
        return list(map(lambda r: self._parse_advice(r.split('|')), result.split("||")))

    def _parse_advice(self, advice):
        return {
            "line_num": advice[0],
            "src_sentence": advice[1],
            "advice": advice[2],
        }


# Testing the class by creating an instance of it
feedback = EssayFeedback()
article = """
I have long sought an opportunity to demonstrate the breadth of my engineering knowledge and experience. As a self-taught engineer, I have developed strong technical skills in electrical design and software development. It gave me a distinctive viewpoint on both the academic and professional spheres of engineering. My approach is characterized by unconventionality, colloquial language, and unorthodox methods that challenge conventional wisdom.

I am motivated by a sincere passion for creativity and problem-solving and an unyielding curiosity regarding the boundaries of science and technology. Throughout my career, I have sought out opportunities to hone my aptitudes by engaging with advanced concepts and tackling difficult challenges. An avid learner, I’ve embraced new technologies, methodologies, and tools with zeal – even when they often present daunting obstacles. By way of such tenacity, I’ve been able to develop complex applications across numerous platforms while refraining from algorithmic fallacies.

A graduate degree would be invaluable in allowing me to further pursue these ambitions. Specifically, being part of the Harvard Graduate School community would allow me to collaborate with industry experts and peers. These people share my dedication and eagerness for growth. Additionally, the vast resources available within the university could facilitate unprecedented levels of exploration into research topics of personal interest. Finally, attendance at Harvard would be advantageous in helping me expand upon my foundational knowledge of electrical engineering. It would ultimately enable me to contribute more substantially to the field.

For all these reasons, I am confident that enrollment in the Harvard Graduate School would offer me a great platform. It would let me realize my goals and help propel me toward future success.
"""
program = "Master of Engineering in Computer Science"
print(feedback.get_advice(article, program))
