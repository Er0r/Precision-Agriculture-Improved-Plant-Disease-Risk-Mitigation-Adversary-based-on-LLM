import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api'

const api = axios.create({
    baseURL: API_BASE_URL,
    timeout: 60000, // 60 seconds timeout
})

// Health check
export const healthCheck = async () => {
    try {
        const response = await api.get('/health/')
        return response.data
    } catch (error) {
        throw new Error(error.response?.data?.message || 'Health check failed')
    }
}

// Upload image
export const uploadImage = async (file, cropType) => {
    try {
        const formData = new FormData()
        formData.append('file', file)
        formData.append('crop_type', cropType)

        const response = await api.post('/upload/', formData, {
            headers: {
                'Content-Type': 'multipart/form-data',
            },
        })

        return response.data
    } catch (error) {
        throw new Error(error.response?.data?.error || 'Upload failed')
    }
}

// Analyze image
export const analyzeImage = async (fileId, cropType) => {
    try {
        const response = await api.post('/analyze/', {
            file_id: fileId,
            crop_type: cropType,
        })

        return response.data
    } catch (error) {
        throw new Error(error.response?.data?.error || 'Analysis failed')
    }
}

// Get analysis history
export const getHistory = async () => {
    try {
        const response = await api.get('/history/')
        return response.data
    } catch (error) {
        throw new Error(error.response?.data?.error || 'Failed to fetch history')
    }
}

// Get sentiment analytics
export const getSentimentAnalytics = async () => {
    try {
        const response = await api.get('/sentiment-analytics/')
        return response.data
    } catch (error) {
        throw new Error(error.response?.data?.error || 'Failed to fetch sentiment analytics')
    }
}

// Analyze text sentiment
export const analyzeSentiment = async (text) => {
    try {
        const response = await api.post('/sentiment-analyze/', { text })
        return response.data
    } catch (error) {
        throw new Error(error.response?.data?.error || 'Sentiment analysis failed')
    }
}

export default api