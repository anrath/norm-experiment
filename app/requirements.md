1. Implementing DocumentService in utils.py:
● Your task is to process the docs/laws.pdf and create a list of Document objects from it.
    ○ Implement the create_documents() method. This involves reading the PDF, parsing the text into meaningful sections, and creating Document objects that encapsulate these sections and their content.
    ○ The parsing logic should accurately identify and separate different laws or sections within the PDF, ensuring the data structure aligns with the Document class requirements.

2. Enhancing QdrantService in utils.py:
● This step requires completing the query method.
● Implement logic to initialize the query engine, run the query with the provided string, and format the results into an Output object containing the query, response, and relevant citations.
● Ensure that self.k, which determines the number of similar vectors to return, is effectively used in the query processing.
● Feel free to complete this section however you would like, but one option is the CitationQueryEngine from llama_index.

3. Setting up the FastAPI endpoint and containerization using Docker:
● Create an API endpoint that accepts a query string and returns a JSON response.
● This endpoint should interact with the QdrantService to process the query and return the results.
● Ensure the output is correctly serialized using the Output class from pydantic.
● Use Docker to containerize the application. Feel free to modify the existing Dockerfile to suit any changes made during development.

4. Running the service:
● Update the README.md to include instructions on how to set up and run the application.
● It is sufficient to tell the user to query the application using the provided OpenAPI documentation, just be sure to clearly describe the required steps.
● Document any assumptions, design choices, and important details that would help in understanding
and evaluating the completed exercise.

Note: This data is being used in a Frontend NextJS app. Think about how the user might want to have the output of the server visualized. Consider any metadata the API could include.