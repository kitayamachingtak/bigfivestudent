class PromptTemplates:

    @staticmethod
    def get_learning_decision_prompt(personality_desc: str, context: dict) -> str:
        history_text = ""
       
        if context.get('previous_action'):
            prev = context['previous_action']
            if prev['action'] == 'self_study':
                history_text = f"In the previous round, you chose to review: {prev['content']}."
            elif prev['action'] == 'ask_teacher':
                history_text = f"In the previous round, you chose to ask the teacher about: {prev['content']}."
            elif prev['action'] == 'skip':
                history_text = "In the previous round, you took a break."
        return f"""[INST] <<SYS>>

{personality_desc}
<</SYS>>

You are learning {context['topic']}.

Decide what to do next. Choose exactly one action:

- "self_study": You want to work on a practice problem.
- "ask_teacher": You want to interact with your teacher.
- "skip": You want to have a break right now.

Your personality should influence your choice. 

Output your decision in JSON format:
{{"action": "self_study/ask_teacher/skip"}}[/INST]"""
    
    
    @staticmethod
    def get_self_study_content_prompt(personality_desc: str, context: dict) -> str:
        return f"""[INST] <<SYS>>
{personality_desc}
<</SYS>>

You decided to self-study in this round.

What specific content or concept do you want to review in {context['topic']}?
Output in JSON format:
{{"content": "the specific topic or concept you want to study"}}[/INST]"""
   
    @staticmethod
    def get_ask_teacher_content_prompt(personality_desc: str, context: dict) -> str:
        return f"""[INST] <<SYS>>
{personality_desc}
<</SYS>>

You decided to ask the teacher for help in this round. Your personality should influence your choice. 

What specific question or concept do you want the teacher to explain in {context['topic']}?
Output in JSON format:
{{"content": "the specific question or concept you want to ask about"}}[/INST]"""
    
    @staticmethod
    def get_teacher_explanation_prompt(student_personality: str, student_question: str, 
                                   retrieved_problem: dict) -> str:
        return f"""[INST] <<SYS>>
You are an experienced mathematics teacher.
<</SYS>>

A student with the following characteristics needs your help:
{student_personality}

The student asks about: {student_question}

Here is a relevant problem from the question bank:
Problem: {retrieved_problem['question']}
Solution: {retrieved_problem['solution']}

Please explain this problem to the student step by step. Focus on:
1. The key concept related to their question
2. How to approach this type of problem
3. Common mistakes to avoid

Keep your explanation clear and concise.[/INST]"""

    @staticmethod
    def get_exam_decision_prompt(personality_desc: str, question: str, context: dict) -> str:
        return f"""[INST] <<SYS>>
{personality_desc}
<</SYS>>

You are taking an exam on {context['topic']}.

Current question:
{question}

You have {context['memory_count']} items in your memory from studying.

You have two options:
1. recall_memory: Try to recall relevant knowledge from your memory before answering
2. answer_directly: Answer the question directly without recalling memory

Output your decision in JSON format:
{{"strategy": "recall_memory/answer_directly"}}[/INST]"""

    @staticmethod
    def get_exam_recall_and_answer_prompt(personality_desc: str, question: str, 
                                          recalled_memory: str = None) -> str:
        memory_section = ""
        if recalled_memory:
            memory_section = f"""
    You are taking an exam. These are your memory from your previous study:
    {recalled_memory}
    
    """
        return f"""[INST] <<SYS>>
    {personality_desc}
    <</SYS>>

    {memory_section}

    ==== EXAM QUESTION (Answer this) ====
    {question}

    Use your memory to solve this exam question. Your personality should influence your thinking process.

    Output format: {{"answer": "put your final answer here"}}[/INST]"""

    @staticmethod
    def get_exam_memory_query_prompt(personality_desc: str, question: str) -> str:
        return f"""[INST] <<SYS>>
{personality_desc}
<</SYS>>

You are taking an exam. Here is the question:
{question}

What topic or concept from your study notes would help you solve this problem?
Think about what you studied that is related to this question. Your personality should influence your choice. 

Output format: {{"query": "describe what you want to recall from your notes"}}[/INST]"""

    @staticmethod
    def get_exam_recall_query_prompt(personality_desc: str, question: str) -> str:
        return f"""[INST] <<SYS>>
{personality_desc}
<</SYS>>

You decided to recall memory before answering. 

Question: {question}

What specific knowledge or concepts do you want to recall from your memory? Your personality should influence your choice. 
Output in JSON format:
{{"query": "the specific knowledge you want to recall"}}[/INST]"""

    @staticmethod
    def get_exam_answer_prompt(personality_desc: str, question: str, 
                               recalled_memory: str = None) -> str:
        memory_text = ""
        if recalled_memory:
            memory_text = f"\nRecalled from your memory:\n{recalled_memory}\n"
        
        return f"""[INST] <<SYS>>
{personality_desc}
<</SYS>>

Question: {question}
{memory_text}
Please provide your answer. Output in JSON format:
{{"answer": "your answer here"}}[/INST]"""