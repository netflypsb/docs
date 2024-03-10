# This is the code for docs 0.2.0
# It is available as a huggingface space at: https://huggingface.co/spaces/netflypsb/docs

import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv
from crewai import Agent, Task, Process, Crew
from langchain_community.tools.pubmed.tool import PubmedQueryRun
from langchain_community.tools.semanticscholar.tool import SemanticScholarQueryRun
from langchain.llms import BaseLLM 
from langchain.agents import AgentExecutor, create_openai_functions_agent
import gradio as gr


# Load environment variables
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Define the llm for agents
llm = ChatOpenAI(
    model="meta-llama/codellama-34b-instruct",
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)

# Define Agents with their specific language model (llm)
def create_agent(role, goal, backstory):
    return Agent(
        role=role,
        goal=goal,
        backstory=backstory,
        verbose=True,
        llm=llm
    )

emergency_physician = create_agent('Emergency Physician', 'Determine emergency care requirements', 'Experienced in rapid assessment and treatment of acute conditions.')
internist = create_agent('Internist', 'Assess and manage internal medicine conditions', 'Specializes in the comprehensive care of adults, managing complex illnesses.')
surgeon = create_agent('Surgeon', 'Evaluate the need for surgical intervention', 'Expert in performing surgical procedures to treat various conditions.')
gynaecologist = create_agent('Gynaecologist', 'Address gynecological aspects of the patient case', 'Focuses on women\'s reproductive health and related surgical treatments.')
obstetrician = create_agent('Obstetrician', 'Consider obstetric care in the patient case', 'Specializes in pregnancy, childbirth, and the postpartum period.')
psychiatrist = create_agent('Psychiatrist', 'Assess mental health aspects of the patient case', 'Expert in the diagnosis, treatment, and prevention of mental illness.')
hospital_director = create_agent('Hospital Director', 'Make final decisions on diagnosis and management plan', 'Oversees the integration of different specialties for optimal patient care.')

# Define Tasks dynamically to include patient case in the description
def create_specialist_task(agent, specialty):
    return Task(
        description=f"Given the patient case, discuss relevant {specialty} aspects.",
        expected_output=(
            "A. Most likely diagnosis\n"
            "B. Most appropriate primary team\n"
            "C. Other treating teams\n"
            "D. Numbered list of treatment plan"
        ),
        agent=agent,
        async_execution=True
    )

tasks = [
    create_specialist_task(emergency_physician, "emergency care"),
    create_specialist_task(internist, "internal medicine"),
    create_specialist_task(surgeon, "surgical intervention"),
    create_specialist_task(gynaecologist, "gynecological care"),
    create_specialist_task(obstetrician, "obstetric care"),
    create_specialist_task(psychiatrist, "mental health"),
]

# Define the final decision task
final_decision_task = Task(
    description="Given all specialist inputs, make the final decision on patient care.",
    expected_output="Final Decision: A. Diagnosis, B. Primary team, C. Treating teams, D. Treatment plan",
    agent=hospital_director,
    context=tasks  # Context from all tasks
)

# Form the Crew with a hierarchical process
crew = Crew(
    agents=[
        emergency_physician,
        internist,
        surgeon,
        gynaecologist,
        obstetrician,
        psychiatrist,
        hospital_director
    ],
    tasks=tasks + [final_decision_task],
    manager_llm=llm,
    process=Process.hierarchical,
    verbose=True,
)

# Gradio UI for Patient Case Input
def kickoff_crew(patient_case):
    # Dynamically update task descriptions with the patient case
    for task in tasks:
        task.description = f"Given the patient case: \"{patient_case}\", {task.description[len('Given the patient case, '):]}"
    final_decision_task.description = f"Given all specialist inputs for the patient case: \"{patient_case}\", make the final decision on patient care."
    results = crew.kickoff()
    return results

iface = gr.Interface(
    fn=kickoff_crew,
    inputs=gr.Textbox(label="Enter Patient Case Here"),
    outputs=gr.Textbox(label="Final Decision"),
    title="Doctor Discussion Crew",
    description="Input a patient case to receive a comprehensive management plan."
)

iface.launch()
