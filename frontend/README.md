# PolicyCortex Frontend

AI-Powered Azure Governance Intelligence Platform - Frontend Application

## Overview

PolicyCortex Frontend is a modern React-based web application built with TypeScript, Material-UI, and Vite. It provides an intuitive interface for managing Azure resources, policies, costs, and compliance through AI-powered insights and real-time monitoring.

## Features

- **Modern React 18** with TypeScript for type safety
- **Material-UI (MUI)** for consistent, accessible design
- **Azure AD Authentication** with MSAL integration
- **Real-time Updates** via WebSocket connections
- **AI-Powered Insights** through conversational interface
- **Responsive Design** for desktop and mobile
- **Dark/Light Theme** support
- **PWA Support** for offline functionality
- **Comprehensive Testing** with Vitest and Testing Library
- **Production-Ready** with Docker containerization

## Architecture

### Technology Stack

- **Frontend Framework**: React 18 with TypeScript
- **Build Tool**: Vite for fast development and building
- **UI Library**: Material-UI (MUI) v5
- **State Management**: Zustand with React Query for server state
- **Authentication**: Azure AD with MSAL
- **Routing**: React Router v6
- **Real-time Communication**: Socket.IO client
- **Charts**: Recharts and MUI X Charts
- **Form Handling**: React Hook Form with Yup validation
- **Testing**: Vitest with React Testing Library
- **Code Quality**: ESLint, Prettier, TypeScript strict mode

### Project Structure

```
src/
├── components/          # Reusable UI components
│   ├── Auth/           # Authentication components
│   ├── Layout/         # Layout components (Header, Sidebar, Footer)
│   ├── UI/            # Generic UI components
│   └── Notifications/ # Notification components
├── pages/             # Page components
│   ├── Dashboard/     # Dashboard pages
│   ├── Policies/      # Policy management pages
│   ├── Resources/     # Resource management pages
│   ├── Costs/         # Cost management pages
│   ├── Conversation/  # AI conversation pages
│   ├── Analytics/     # Analytics pages
│   ├── Security/      # Security pages
│   ├── Settings/      # Settings pages
│   └── Auth/          # Authentication pages
├── hooks/             # Custom React hooks
├── services/          # API service layer
├── store/             # State management (Zustand stores)
├── types/             # TypeScript type definitions
├── config/            # Configuration files
├── utils/             # Utility functions
├── assets/            # Static assets
└── test/              # Test utilities and setup
```

## Prerequisites

- **Node.js** 18.0.0 or higher
- **npm** 9.0.0 or higher (or yarn/pnpm)
- **Azure AD Application** with appropriate permissions
- **Backend API** (PolicyCortex backend services)

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/policycortex.git
cd policycortex/frontend
```

### 2. Install Dependencies

```bash
npm install
```

### 3. Environment Configuration

Copy the example environment file and configure it:

```bash
cp .env.example .env
```

Update the `.env` file with your configuration:

```env
# Azure AD Configuration
VITE_AZURE_CLIENT_ID=your-azure-client-id
VITE_AZURE_TENANT_ID=your-azure-tenant-id
VITE_AZURE_REDIRECT_URI=http://localhost:3000

# API Configuration
VITE_API_BASE_URL=http://localhost:8000/api
VITE_WS_URL=ws://localhost:8000/ws

# Feature Flags
VITE_ENABLE_ANALYTICS=true
VITE_ENABLE_NOTIFICATIONS=true
VITE_ENABLE_WEBSOCKET=true
VITE_ENABLE_PWA=true
VITE_ENABLE_DARK_MODE=true
```

### 4. Azure AD Setup

1. Register a new application in Azure AD
2. Configure redirect URIs for your domain
3. Grant necessary API permissions:
   - `User.Read` (Microsoft Graph)
   - `https://management.azure.com/user_impersonation`
4. Update the environment variables with your application details

### 5. Start Development Server

```bash
npm run dev
```

The application will be available at `http://localhost:3000`

## Available Scripts

### Development

```bash
# Start development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview

# Type checking
npm run type-check
```

### Code Quality

```bash
# Lint code
npm run lint

# Fix linting issues
npm run lint:fix

# Format code
npm run format

# Check formatting
npm run format:check
```

### Testing

```bash
# Run tests
npm run test

# Run tests with UI
npm run test:ui

# Run tests with coverage
npm run test:coverage

# Run tests in watch mode
npm run test:watch
```

## Docker Deployment

### Build Docker Image

```bash
docker build -t policycortex-frontend .
```

### Run with Docker

```bash
docker run -p 80:80 \
  -e VITE_AZURE_CLIENT_ID=your-client-id \
  -e VITE_AZURE_TENANT_ID=your-tenant-id \
  -e VITE_API_BASE_URL=https://api.yourcompany.com/api \
  -e VITE_WS_URL=wss://api.yourcompany.com/ws \
  policycortex-frontend
```

### Docker Compose

```yaml
version: '3.8'
services:
  frontend:
    build: .
    ports:
      - "80:80"
    environment:
      - VITE_AZURE_CLIENT_ID=your-client-id
      - VITE_AZURE_TENANT_ID=your-tenant-id
      - VITE_API_BASE_URL=http://backend:8000/api
      - VITE_WS_URL=ws://backend:8000/ws
    depends_on:
      - backend
```

## Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `VITE_AZURE_CLIENT_ID` | Azure AD Application Client ID | - | Yes |
| `VITE_AZURE_TENANT_ID` | Azure AD Tenant ID | - | Yes |
| `VITE_AZURE_REDIRECT_URI` | Azure AD Redirect URI | `http://localhost:3000` | No |
| `VITE_API_BASE_URL` | Backend API Base URL | `http://localhost:8000/api` | No |
| `VITE_WS_URL` | WebSocket Server URL | `ws://localhost:8000/ws` | No |
| `VITE_ENABLE_ANALYTICS` | Enable analytics features | `true` | No |
| `VITE_ENABLE_NOTIFICATIONS` | Enable notifications | `true` | No |
| `VITE_ENABLE_WEBSOCKET` | Enable WebSocket features | `true` | No |
| `VITE_ENABLE_PWA` | Enable PWA features | `true` | No |
| `VITE_ENABLE_DARK_MODE` | Enable dark mode | `true` | No |

### Feature Flags

The application supports various feature flags to enable/disable functionality:

- **Analytics**: User behavior tracking and analytics
- **Notifications**: Real-time notifications system
- **WebSocket**: Real-time data updates
- **PWA**: Progressive Web App features
- **Dark Mode**: Dark theme support

## Authentication

The application uses Azure AD for authentication via MSAL (Microsoft Authentication Library). Users are required to authenticate before accessing the application.

### Supported Authentication Flows

- **Authorization Code Flow with PKCE** (recommended)
- **Implicit Flow** (fallback for older browsers)

### Permissions

The application requires the following Azure AD permissions:

- `User.Read` - Read user profile information
- `https://management.azure.com/user_impersonation` - Access Azure resources

## API Integration

The frontend communicates with the PolicyCortex backend API for:

- **Authentication**: User authentication and authorization
- **Policies**: Policy management and compliance
- **Resources**: Azure resource inventory and management
- **Costs**: Cost analysis and budgeting
- **Notifications**: Real-time alerts and notifications
- **Analytics**: Usage analytics and insights

### API Client

The application uses a custom API client with the following features:

- **Automatic Token Refresh**: Handles token renewal automatically
- **Request/Response Interceptors**: Logging and error handling
- **Retry Logic**: Automatic retry for failed requests
- **Loading States**: Integrated loading state management

## Testing

### Test Structure

```
src/
├── __tests__/          # Test files
├── components/
│   └── Component.test.tsx
├── hooks/
│   └── useHook.test.ts
├── services/
│   └── service.test.ts
└── utils/
    └── util.test.ts
```

### Testing Strategy

- **Unit Tests**: Individual component and function testing
- **Integration Tests**: API integration and service testing
- **End-to-End Tests**: Full user flow testing (planned)

### Test Utilities

The project includes comprehensive test utilities:

- **Custom Render**: Renders components with providers
- **Mock Data**: Standardized mock data for testing
- **API Mocking**: MSW for API mocking
- **User Interactions**: Testing Library user event utilities

## Performance

### Optimization Strategies

- **Code Splitting**: Route-based code splitting
- **Lazy Loading**: Lazy loading of components and images
- **Caching**: React Query for server state caching
- **Bundle Analysis**: Webpack bundle analyzer integration
- **PWA Features**: Service worker for offline functionality

### Performance Monitoring

- **Web Vitals**: Core Web Vitals monitoring
- **Error Tracking**: Sentry integration (optional)
- **Analytics**: Google Analytics integration (optional)

## Security

### Security Features

- **Content Security Policy**: CSP headers in production
- **HTTPS Only**: Secure communication in production
- **Token Security**: Secure token storage and handling
- **Input Validation**: Client-side input validation
- **XSS Protection**: XSS prevention measures

### Security Best Practices

- Regular dependency updates
- Security audit with `npm audit`
- Environment variable protection
- Secure build process

## Deployment

### Production Build

```bash
npm run build
```

### Deployment Options

1. **Static Hosting**: Deploy to Netlify, Vercel, or similar
2. **Container Deployment**: Use Docker for containerized deployment
3. **CDN Deployment**: Deploy to Azure Static Web Apps or AWS S3

### Health Check

The application includes a health check endpoint at `/health` for monitoring.

## Troubleshooting

### Common Issues

1. **Authentication Errors**
   - Verify Azure AD configuration
   - Check redirect URI configuration
   - Ensure proper permissions are granted

2. **API Connection Issues**
   - Verify backend API is running
   - Check CORS configuration
   - Verify API base URL

3. **Build Errors**
   - Clear node_modules and reinstall
   - Check TypeScript configuration
   - Verify environment variables

### Debug Mode

Enable debug mode for detailed logging:

```env
VITE_ENABLE_DEBUG=true
VITE_LOG_LEVEL=debug
```

## Contributing

### Development Workflow

1. Create a feature branch
2. Make your changes
3. Run tests and linting
4. Submit a pull request

### Code Standards

- Follow TypeScript best practices
- Use React hooks and functional components
- Write comprehensive tests
- Follow the established project structure

### Commit Messages

Follow conventional commit format:

```
feat: add new dashboard component
fix: resolve authentication issue
docs: update README with new features
test: add tests for policy service
```

## Support

For support and questions:

- **Issues**: GitHub Issues
- **Documentation**: Project Wiki
- **Email**: support@policycortex.com

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for a list of changes and updates.

---

**PolicyCortex Frontend** - AI-Powered Azure Governance Intelligence Platform