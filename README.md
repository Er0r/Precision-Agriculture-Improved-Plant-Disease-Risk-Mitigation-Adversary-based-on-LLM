# 🌾 Smart Crop Disease Analyzer

Modern **React + Django** application for AI-powered crop disease detection and treatment recommendations using machine learning models and NVIDIA NIM LLM integration.

## ✨ Features

- **⚛️ Modern React UI**: Built with React 18, Vite, and Tailwind CSS
- **🏗️ Django REST API**: Robust backend with Django REST Framework
- **🖼️ Smart Image Upload**: Drag & drop interface with real-time preview
- **🌾 Multi-Crop Support**: Rice and Jute disease analysis
- **🤖 AI-Powered Analysis**: ML disease detection + Expert LLM recommendations
- **💊 Detailed Treatment Plans**: Specific fungicides, dosages, and timelines
- **🛡️ Prevention Strategies**: Proactive disease management advice
- **📊 Risk Assessment**: Economic impact and confidence scoring
- **📱 Responsive Design**: Beautiful UI that works on all devices
- **💾 Database Storage**: Persistent storage of images and analysis results

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    REACT FRONTEND                           │
│  Modern UI with Vite + Tailwind CSS (Port 3000)           │
└─────────────────────┬───────────────────────────────────────┘
                      │ HTTP API Calls
┌─────────────────────▼───────────────────────────────────────┐
│                DJANGO REST API                              │
│  RESTful API with DRF + CORS (Port 8000)                  │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│              MCP + NIM INTEGRATION                          │
│  Disease Detection + Expert Recommendations                │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### **Backend Setup (Django)**
```bash
cd backend
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

### **Frontend Setup (React)**
```bash
cd frontend
npm install
npm run dev
```

### **Access Application**
- **Frontend**: `http://localhost:3000` (React UI)
- **Backend API**: `http://localhost:8000` (Django API)
- **Admin Panel**: `http://localhost:8000/admin` (Django Admin)

## 📁 Project Structure

```
├── 🎨 React Frontend (frontend/)
│   ├── src/
│   │   ├── components/          # React components
│   │   │   ├── ImageUpload.jsx  # Upload interface
│   │   │   ├── AnalysisResults.jsx # Results display
│   │   │   ├── LoadingOverlay.jsx  # Loading states
│   │   │   └── Toast.jsx        # Notifications
│   │   ├── services/
│   │   │   └── api.js          # API client
│   │   ├── App.jsx             # Main app component
│   │   └── main.jsx            # React entry point
│   ├── package.json            # Node dependencies
│   ├── vite.config.js          # Vite configuration
│   └── tailwind.config.js      # Tailwind configuration
│
├── 🔧 Django Backend (backend/)
│   ├── crop_disease_api/       # Django project
│   │   ├── settings.py         # Django settings
│   │   ├── urls.py            # URL routing
│   │   └── wsgi.py            # WSGI config
│   ├── analysis/              # Django app
│   │   ├── models.py          # Database models
│   │   ├── views.py           # API views
│   │   ├── serializers.py     # DRF serializers
│   │   ├── urls.py            # App URLs
│   │   ├── admin.py           # Admin interface
│   │   └── mcp_integration.py # AI integration
│   ├── manage.py              # Django management
│   ├── requirements.txt       # Python dependencies
│   └── .env.example          # Environment template
│
└── 📚 Documentation
    └── README.md              # This file
```

## 🔧 Configuration

### **Backend Configuration**
Create `backend/.env` file:
```env
# Django Configuration
SECRET_KEY=your-secret-key-here
DEBUG=True

# NVIDIA NIM LLM Configuration (Optional)
NIM_BASE_URL=https://integrate.api.nvidia.com/v1
NIM_API_KEY=your-api-key-here
NIM_MODEL=openai/gpt-oss-120b
```

### **Frontend Configuration**
The frontend automatically proxies API calls to Django backend.

## 🎯 How It Works

1. **📸 Upload**: Drag & drop crop image in React UI
2. **📡 API Call**: Frontend sends image to Django backend
3. **💾 Storage**: Django saves image and metadata to database
4. **🔍 Detect**: AI identifies disease with confidence score
5. **🤖 Analyze**: NIM LLM generates expert treatment plan
6. **💾 Save**: Analysis results stored in database
7. **📋 Results**: React displays detailed recommendations

## 🌾 Supported Crops & Diseases

### **Rice**
- Brown Spot, Rice Blast, Bacterial Leaf Blight, Tungro, Healthy

### **Jute**
- Anthracnose, Stem Rot, Healthy

## 🔧 System Requirements

- **Python**: 3.8+
- **Node.js**: 16+
- **Memory**: 4GB RAM minimum
- **Storage**: 2GB free space
- **Network**: Internet connection for NIM LLM (optional)

## 🛠️ Development Commands

### **Backend (Django)**
```bash
cd backend

# Database operations
python manage.py makemigrations
python manage.py migrate

# Create superuser
python manage.py createsuperuser

# Run development server
python manage.py runserver

# Run tests
python manage.py test
```

### **Frontend (React)**
```bash
cd frontend

# Development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

## 📡 API Endpoints

- `GET /api/health/` - Health check
- `POST /api/upload/` - Upload image
- `POST /api/analyze/` - Analyze image
- `GET /api/history/` - Get analysis history
- `GET /api/images/<uuid>/` - Get image details
- `GET /admin/` - Django admin interface

## 🛠️ Troubleshooting

**Common Issues:**
- **Port conflicts**: Django runs on 8000, React on 3000
- **CORS errors**: Ensure django-cors-headers is configured
- **Database errors**: Run `python manage.py migrate`
- **MCP not found**: Ensure `../mcp/` directory exists
- **Upload fails**: Check file size (<16MB) and format

## 🎉 Production Ready Features

✅ **Modern React architecture** with Vite and Tailwind CSS  
✅ **Django REST API** with proper authentication support  
✅ **Database persistence** with SQLite (easily upgradeable to PostgreSQL)  
✅ **Admin interface** for managing data  
✅ **Inference-only models** (no training overhead)  
✅ **Comprehensive recommendations** with specific products  
✅ **Beautiful, responsive interface** for all devices  
✅ **Robust error handling** and fallbacks  

## 🚀 Deployment

### **Backend Deployment**
- Use PostgreSQL for production database
- Configure static file serving
- Set up proper environment variables
- Use Gunicorn or uWSGI for production server

### **Frontend Deployment**
- Build with `npm run build`
- Serve static files with nginx or similar
- Configure API base URL for production 
