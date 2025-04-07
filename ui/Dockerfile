FROM node:18-alpine

WORKDIR /app

# Copy package files and install dependencies
COPY package*.json ./
RUN npm install

# Copy the rest of the application
COPY src/ ./src/
COPY public/ ./public/
COPY server.js ./

# Expose ports - 9029 for API, 3000 for React dev server
EXPOSE 9029 3000

# Start both the API server and React dev server
CMD ["sh", "-c", "node server.js & BROWSER=none PORT=3000 npm run dev"] 