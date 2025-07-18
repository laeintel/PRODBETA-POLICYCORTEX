import { Avatar, AvatarProps } from '@mui/material'
import { User } from '@/types'

interface UserAvatarProps extends Omit<AvatarProps, 'src' | 'alt'> {
  user: User | null
  size?: number
}

export const UserAvatar = ({ user, size = 40, ...props }: UserAvatarProps) => {
  const getInitials = (user: User | null) => {
    if (!user) return '?'
    
    const firstName = user.firstName || ''
    const lastName = user.lastName || ''
    
    if (firstName && lastName) {
      return `${firstName[0]}${lastName[0]}`.toUpperCase()
    }
    
    if (firstName) {
      return firstName[0].toUpperCase()
    }
    
    if (user.email) {
      return user.email[0].toUpperCase()
    }
    
    return '?'
  }

  const getDisplayName = (user: User | null) => {
    if (!user) return 'Unknown User'
    
    if (user.displayName) return user.displayName
    if (user.firstName && user.lastName) return `${user.firstName} ${user.lastName}`
    if (user.firstName) return user.firstName
    if (user.email) return user.email
    
    return 'Unknown User'
  }

  return (
    <Avatar
      src={user?.avatar}
      alt={getDisplayName(user)}
      sx={{
        width: size,
        height: size,
        ...props.sx,
      }}
      {...props}
    >
      {getInitials(user)}
    </Avatar>
  )
}