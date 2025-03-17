# CIFAR-10 Image Classifier Frontend

A modern React-based web interface for the CIFAR-10 image classification service. This application provides an intuitive user interface for uploading images and getting real-time predictions using a trained deep learning model.

## Features

- 🖼️ Drag-and-drop image upload
- 👁️ Real-time image preview
- 🔄 Live prediction results
- 📊 Confidence score visualization
- 🎨 Material-UI based modern design
- 📱 Responsive layout for all devices

## Technologies Used

- React 18 with TypeScript
- Material-UI (MUI) for styling
- Axios for API communication
- Docker for containerization

## Getting Started

### Prerequisites

- Node.js 18.x or higher
- npm 8.x or higher
- Docker and Docker Compose (if running with containers)

### Local Development

1. Install dependencies:
   ```bash
   npm install
   ```

2. Create a `.env` file in the root directory:
   ```env
   REACT_APP_API_URL=http://localhost:8000
   ```

3. Start the development server:
   ```bash
   npm start
   ```

The application will be available at `http://localhost:3000`.

### Docker Deployment

The application can be run as part of the complete stack using Docker Compose:

```bash
docker-compose up --build
```

## Project Structure

```
frontend/
├── src/
│   ├── components/     # React components
│   │   └── ImageUpload.tsx
│   ├── App.tsx        # Main application component
│   └── index.tsx      # Application entry point
├── public/            # Static assets
├── Dockerfile         # Docker configuration
└── package.json       # Project dependencies
```

## Component Documentation

### ImageUpload

The main component that handles:
- Image file selection and preview
- Communication with the backend API
- Display of prediction results
- Loading states and error handling

Props: None

State:
- `selectedImage`: Currently selected image file
- `previewUrl`: URL for image preview
- `loading`: Loading state during API calls
- `result`: Prediction results from the API
- `error`: Error state for failed operations

## API Integration

The frontend communicates with the backend API at `REACT_APP_API_URL`. The main endpoint used is:

- `POST /predict`: Sends image data and receives classification results
  ```typescript
  // Response format
  interface InferenceResult {
    class_name: string;    // Predicted class name
    confidence: number;    // Confidence score (0-1)
  }
  ```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| REACT_APP_API_URL | Backend API URL | http://localhost:8000 |

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License - feel free to use this code for your own projects.
