# frontend/Dockerfile
FROM node:20-alpine

WORKDIR /app

# Dependencies copy aur install karo
COPY package*.json ./
RUN npm install

# Baaki code copy karo
COPY . .

# Vite server ko expose karo
EXPOSE 5176

# Server start karo
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "5176"]