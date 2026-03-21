#use lightweight python image
FROM python:3.10-slim

#set working directory
WORKDIR /app

#copy project files
COPY . .

#Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

#Expose port (Render uses 10000)
EXPOSE 10000

#start FastAPI app
CMD [ "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "10000"]

