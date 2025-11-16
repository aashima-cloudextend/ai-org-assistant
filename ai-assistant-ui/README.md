# 🤖 CEIRA - CloudExtend Intelligence & Response Assistant

## Futuristic UI for AI Organization Assistant

A stunning, modern React interface for CloudExtend's AI-powered knowledge assistant. Features glassmorphism design, animated gradients, and a cyberpunk-inspired color scheme.

## 🎨 Design Features

- **Animated Background**: Dynamic gradient orbs and animated grid overlay
- **Cursor Glow Effect**: Interactive mouse-tracking glow effect
- **Glassmorphism**: Frosted glass effect with backdrop blur
- **Neon Accents**: Cyan (#00f2ff), Purple (#7b2ff7), and Pink (#ff2e97) color scheme
- **CloudExtend Branding**: Official CloudExtend blue and teal colors
- **Smooth Animations**: Hover effects, transitions, and loading states
- **Responsive Design**: Works beautifully on desktop, tablet, and mobile

## 🚀 Quick Start

### Prerequisites

- Node.js 14+ and npm
- Backend server running on `http://localhost:8000`

### Installation

```bash
cd ai-assistant-ui
npm install
```

### Development Mode

```bash
npm start
```

The app will open at `http://localhost:3000`

### Production Build

```bash
npm run build
```

## 🎯 Features

### 1. Query CEIRA Intelligence
- **Natural Language Queries**: Ask questions in plain English
- **Role-Based Responses**: Tailored answers for Developers, Support, Managers, or General users
- **Source Citations**: View relevant documents with similarity scores
- **Confidence Scores**: See how confident CEIRA is in each response
- **Processing Time**: Real-time performance metrics

### 2. Sync Knowledge Base
- **Multi-Source Support**: GitHub, Confluence, and Jira integration
- **Path Filtering**: Include/exclude specific paths
- **Real-Time Status**: Live updates during sync process
- **Progress Tracking**: Monitor documents, chunks, and errors

## 🔧 Configuration

Update the API endpoint in the source files if your backend is not on `localhost:8000`:

```javascript
// In QueryForm.js and SyncForm.js
const response = await fetch('http://YOUR_BACKEND_URL/query', {
  // ...
});
```

## 🎨 Customization

### Color Scheme

Edit CSS variables in `App.css`:

```css
:root {
  --primary-cyan: #00f2ff;
  --primary-purple: #7b2ff7;
  --primary-pink: #ff2e97;
  --cloudextend-blue: #0066cc;
  --cloudextend-teal: #00b4d8;
}
```

### Fonts

The UI uses:
- **Orbitron**: Futuristic headings and titles
- **Inter**: Clean, modern body text

## 📱 Browser Support

- Chrome/Edge 90+
- Firefox 88+
- Safari 14+

## 🐛 Troubleshooting

### Backend Connection Issues

If you see "Failed to connect to server" errors:

1. Ensure the backend is running: `cd .. && python main.py`
2. Check CORS settings in the backend
3. Verify the API URL in fetch calls

### Styling Issues

If styles don't load correctly:

1. Clear browser cache
2. Delete `node_modules` and reinstall: `rm -rf node_modules && npm install`
3. Check browser console for errors

## 🏗️ Tech Stack

- **React 18**: Modern UI framework
- **CSS3**: Custom animations and effects
- **Google Fonts**: Orbitron and Inter
- **Fetch API**: REST communication with backend

## 📄 License

Part of the AI Organization Assistant project by CloudExtend.

## 🤝 Contributing

To contribute:

1. Create a feature branch
2. Make your changes
3. Test thoroughly on different screen sizes
4. Submit a pull request

---

**Built with ❤️ for CloudExtend**

*CEIRA - Your intelligent assistant for organizational knowledge*
