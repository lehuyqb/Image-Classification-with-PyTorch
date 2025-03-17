import React, { useState } from 'react';
import { 
  Box, 
  Button, 
  Card, 
  CardContent, 
  CircularProgress, 
  Typography,
  Container
} from '@mui/material';
import axios from 'axios';

interface InferenceResult {
  class_name: string;
  confidence: number;
}

const ImageUpload: React.FC = () => {
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<InferenceResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleImageSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedImage(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResult(null);
      setError(null);
    }
  };

  const handleUpload = async () => {
    if (!selectedImage) return;

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', selectedImage);

    try {
      const response = await axios.post(`${process.env.REACT_APP_API_URL}/predict`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setResult(response.data);
    } catch (err) {
      setError('Failed to get prediction. Please try again.');
      console.error('Error:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container maxWidth="sm">
      <Box sx={{ mt: 4 }}>
        <Card>
          <CardContent>
            <Typography variant="h5" component="h2" gutterBottom>
              Image Classification
            </Typography>

            <Box sx={{ mb: 2 }}>
              <input
                accept="image/*"
                style={{ display: 'none' }}
                id="image-upload"
                type="file"
                onChange={handleImageSelect}
              />
              <label htmlFor="image-upload">
                <Button variant="contained" component="span">
                  Select Image
                </Button>
              </label>
            </Box>

            {previewUrl && (
              <Box sx={{ mb: 2 }}>
                <img
                  src={previewUrl}
                  alt="Preview"
                  style={{ maxWidth: '100%', maxHeight: '300px' }}
                />
              </Box>
            )}

            {selectedImage && (
              <Button
                variant="contained"
                color="primary"
                onClick={handleUpload}
                disabled={loading}
                sx={{ mb: 2 }}
              >
                {loading ? <CircularProgress size={24} /> : 'Get Prediction'}
              </Button>
            )}

            {error && (
              <Typography color="error" sx={{ mb: 2 }}>
                {error}
              </Typography>
            )}

            {result && (
              <Box sx={{ mt: 2 }}>
                <Typography variant="h6">Result:</Typography>
                <Typography>
                  Class: {result.class_name}
                </Typography>
                <Typography>
                  Confidence: {(result.confidence * 100).toFixed(2)}%
                </Typography>
              </Box>
            )}
          </CardContent>
        </Card>
      </Box>
    </Container>
  );
};

export default ImageUpload; 