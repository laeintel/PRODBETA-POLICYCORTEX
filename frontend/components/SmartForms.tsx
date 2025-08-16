"use client"

import { useState, useEffect, useCallback, useRef } from 'react'
import { useForm, Controller, FieldValues, FieldError } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import { z } from 'zod'
import { Save, AlertCircle, CheckCircle2, Clock, Loader2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Checkbox } from '@/components/ui/checkbox'
import { Switch } from '@/components/ui/switch'
import { Textarea } from '@/components/ui/textarea'
import { cn } from '@/lib/utils'

// Form field configuration interface
export interface FormFieldConfig {
  name: string
  label: string
  type: 'text' | 'email' | 'password' | 'number' | 'select' | 'multiselect' | 'checkbox' | 'switch' | 'textarea' | 'file' | 'date'
  placeholder?: string
  description?: string
  required?: boolean
  validation?: z.ZodSchema
  options?: Array<{ value: string; label: string }>
  dependencies?: string[] // Fields this depends on
  conditional?: (values: FieldValues) => boolean // Show/hide based on other fields
  autoComplete?: string
  formatValue?: (value: any) => string
  parseValue?: (value: string) => any
}

// Smart form configuration
export interface SmartFormConfig {
  id: string
  title: string
  description?: string
  fields: FormFieldConfig[]
  schema: z.ZodSchema
  autoSave?: boolean
  autoSaveDelay?: number
  submitText?: string
  showProgress?: boolean
  templates?: FormTemplate[]
}

// Form template interface
export interface FormTemplate {
  id: string
  name: string
  description?: string
  values: FieldValues
}

// Auto-save hook
function useAutoSave<T extends FieldValues>(
  data: T,
  onSave: (data: T) => Promise<void>,
  delay: number = 2000,
  enabled: boolean = true
) {
  const [isSaving, setIsSaving] = useState(false)
  const [lastSaved, setLastSaved] = useState<Date | null>(null)
  const timeoutRef = useRef<NodeJS.Timeout>()

  const saveData = useCallback(async (data: T) => {
    if (!enabled) return

    setIsSaving(true)
    try {
      await onSave(data)
      setLastSaved(new Date())
    } catch (error) {
      console.error('Auto-save failed:', error)
    } finally {
      setIsSaving(false)
    }
  }, [onSave, enabled])

  useEffect(() => {
    if (!enabled) return

    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current)
    }

    timeoutRef.current = setTimeout(() => {
      saveData(data)
    }, delay)

    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current)
      }
    }
  }, [data, delay, enabled, saveData])

  return { isSaving, lastSaved }
}

// Field validation display component
function FieldValidation({ error, isValid }: { error?: FieldError; isValid?: boolean }) {
  if (!error && !isValid) return null

  return (
    <div className="flex items-center gap-1 mt-1">
      {error ? (
        <>
          <AlertCircle className="h-3 w-3 text-destructive" />
          <span className="text-xs text-destructive">{error.message}</span>
        </>
      ) : isValid ? (
        <>
          <CheckCircle2 className="h-3 w-3 text-green-500" />
          <span className="text-xs text-green-600">Valid</span>
        </>
      ) : null}
    </div>
  )
}

// Smart input component with real-time validation
interface SmartInputProps {
  config: FormFieldConfig
  value: any
  onChange: (value: any) => void
  error?: FieldError
  disabled?: boolean
}

function SmartInput({ config, value, onChange, error, disabled }: SmartInputProps) {
  const [isValid, setIsValid] = useState(false)
  const [validationMessage, setValidationMessage] = useState<string>('')

  // Real-time validation
  useEffect(() => {
    if (config.validation && value) {
      try {
        config.validation.parse(value)
        setIsValid(true)
        setValidationMessage('')
      } catch (validationError) {
        setIsValid(false)
        if (validationError instanceof z.ZodError) {
          setValidationMessage(validationError.issues[0]?.message || 'Invalid value')
        }
      }
    }
  }, [value, config.validation])

  const handleChange = (newValue: any) => {
    const parsed = config.parseValue ? config.parseValue(newValue) : newValue
    onChange(parsed)
  }

  const displayValue = config.formatValue ? config.formatValue(value) : value

  switch (config.type) {
    case 'select':
      return (
        <div className="space-y-1">
          <Label htmlFor={config.name} className="flex items-center gap-1">
            {config.label}
            {config.required && <span className="text-destructive">*</span>}
          </Label>
          <Select value={displayValue} onChange={(e) => handleChange(e.target.value)} disabled={disabled}>
            {config.options?.map(option => (
              <SelectItem key={option.value} value={option.value}>
                {option.label}
              </SelectItem>
            ))}
          </Select>
          {config.description && (
            <p className="text-xs text-muted-foreground">{config.description}</p>
          )}
          <FieldValidation error={error} isValid={isValid && !error} />
        </div>
      )

    case 'checkbox':
      return (
        <div className="space-y-1">
          <div className="flex items-center space-x-2">
            <Checkbox
              id={config.name}
              checked={value}
              onCheckedChange={onChange}
              disabled={disabled}
            />
            <Label htmlFor={config.name} className="flex items-center gap-1">
              {config.label}
              {config.required && <span className="text-destructive">*</span>}
            </Label>
          </div>
          {config.description && (
            <p className="text-xs text-muted-foreground ml-6">{config.description}</p>
          )}
          <FieldValidation error={error} isValid={isValid && !error} />
        </div>
      )

    case 'switch':
      return (
        <div className="space-y-1">
          <div className="flex items-center justify-between">
            <div>
              <Label htmlFor={config.name} className="flex items-center gap-1">
                {config.label}
                {config.required && <span className="text-destructive">*</span>}
              </Label>
              {config.description && (
                <p className="text-xs text-muted-foreground">{config.description}</p>
              )}
            </div>
            <Switch
              id={config.name}
              checked={value}
              onCheckedChange={onChange}
              disabled={disabled}
            />
          </div>
          <FieldValidation error={error} isValid={isValid && !error} />
        </div>
      )

    case 'textarea':
      return (
        <div className="space-y-1">
          <Label htmlFor={config.name} className="flex items-center gap-1">
            {config.label}
            {config.required && <span className="text-destructive">*</span>}
          </Label>
          <Textarea
            id={config.name}
            value={displayValue || ''}
            onChange={(e) => handleChange(e.target.value)}
            placeholder={config.placeholder}
            disabled={disabled}
            className={cn(error && "border-destructive")}
            autoComplete={config.autoComplete}
          />
          {config.description && (
            <p className="text-xs text-muted-foreground">{config.description}</p>
          )}
          <FieldValidation error={error} isValid={isValid && !error} />
        </div>
      )

    default:
      return (
        <div className="space-y-1">
          <Label htmlFor={config.name} className="flex items-center gap-1">
            {config.label}
            {config.required && <span className="text-destructive">*</span>}
          </Label>
          <Input
            id={config.name}
            type={config.type}
            value={displayValue || ''}
            onChange={(e) => handleChange(e.target.value)}
            placeholder={config.placeholder}
            disabled={disabled}
            className={cn(error && "border-destructive")}
            autoComplete={config.autoComplete}
          />
          {config.description && (
            <p className="text-xs text-muted-foreground">{config.description}</p>
          )}
          <FieldValidation error={error} isValid={isValid && !error} />
        </div>
      )
  }
}

// Form progress indicator
function FormProgress({ fields, values, errors }: { 
  fields: FormFieldConfig[]
  values: FieldValues
  errors: Record<string, FieldError>
}) {
  const requiredFields = fields.filter(f => f.required)
  const completedFields = requiredFields.filter(f => 
    values[f.name] && !errors[f.name]
  )
  
  const progress = requiredFields.length > 0 
    ? (completedFields.length / requiredFields.length) * 100 
    : 100

  return (
    <div className="space-y-2">
      <div className="flex justify-between text-sm">
        <span>Form Progress</span>
        <span>{Math.round(progress)}% Complete</span>
      </div>
      <div className="h-2 bg-muted rounded-full">
        <div 
          className="h-2 bg-primary rounded-full transition-all"
          style={{ width: `${progress}%` }}
        />
      </div>
      <div className="text-xs text-muted-foreground">
        {completedFields.length} of {requiredFields.length} required fields completed
      </div>
    </div>
  )
}

// Auto-save status indicator
function AutoSaveStatus({ isSaving, lastSaved }: { isSaving: boolean; lastSaved: Date | null }) {
  if (isSaving) {
    return (
      <div className="flex items-center gap-1 text-xs text-muted-foreground">
        <Loader2 className="h-3 w-3 animate-spin" />
        Saving...
      </div>
    )
  }

  if (lastSaved) {
    return (
      <div className="flex items-center gap-1 text-xs text-muted-foreground">
        <Save className="h-3 w-3" />
        Saved {lastSaved.toLocaleTimeString()}
      </div>
    )
  }

  return (
    <div className="flex items-center gap-1 text-xs text-muted-foreground">
      <Clock className="h-3 w-3" />
      Auto-save enabled
    </div>
  )
}

// Main smart form component
interface SmartFormProps {
  config: SmartFormConfig
  initialValues?: FieldValues
  onSubmit: (data: FieldValues) => Promise<void>
  onAutoSave?: (data: FieldValues) => Promise<void>
  onTemplateApply?: (template: FormTemplate) => void
  className?: string
}

export function SmartForm({
  config,
  initialValues = {},
  onSubmit,
  onAutoSave,
  onTemplateApply,
  className
}: SmartFormProps) {
  const [isSubmitting, setIsSubmitting] = useState(false)
  
  const {
    control,
    handleSubmit,
    watch,
    formState: { errors, isValid },
    setValue,
    reset
  } = useForm({
    resolver: config.schema ? zodResolver(config.schema as any) : undefined,
    defaultValues: initialValues,
    mode: 'onChange'
  })

  const watchedValues = watch()
  
  const { isSaving, lastSaved } = useAutoSave(
    watchedValues,
    onAutoSave || (() => Promise.resolve()),
    config.autoSaveDelay || 2000,
    config.autoSave || false
  )

  // Filter fields based on conditional logic
  const visibleFields = config.fields.filter(field => 
    !field.conditional || field.conditional(watchedValues)
  )

  const handleFormSubmit = async (data: FieldValues) => {
    setIsSubmitting(true)
    try {
      await onSubmit(data)
    } catch (error) {
      console.error('Form submission failed:', error)
    } finally {
      setIsSubmitting(false)
    }
  }

  const applyTemplate = (template: FormTemplate) => {
    Object.entries(template.values).forEach(([field, value]) => {
      setValue(field, value)
    })
    onTemplateApply?.(template)
  }

  return (
    <Card className={className}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle>{config.title}</CardTitle>
            {config.description && (
              <p className="text-sm text-muted-foreground mt-1">{config.description}</p>
            )}
          </div>
          
          {config.autoSave && (
            <AutoSaveStatus isSaving={isSaving} lastSaved={lastSaved} />
          )}
        </div>
        
        {config.showProgress && (
          <FormProgress 
            fields={config.fields} 
            values={watchedValues} 
            errors={errors as Record<string, FieldError>} 
          />
        )}
      </CardHeader>
      
      <CardContent className="space-y-6">
        {/* Form templates */}
        {config.templates && config.templates.length > 0 && (
          <div className="space-y-2">
            <Label>Form Templates</Label>
            <div className="flex flex-wrap gap-2">
              {config.templates.map(template => (
                <Button
                  key={template.id}
                  variant="outline"
                  size="sm"
                  onClick={() => applyTemplate(template)}
                  className="text-xs"
                >
                  {template.name}
                </Button>
              ))}
            </div>
          </div>
        )}

        <form onSubmit={handleSubmit(handleFormSubmit)} className="space-y-4">
          {visibleFields.map(field => (
            <Controller
              key={field.name}
              name={field.name}
              control={control}
              render={({ field: { value, onChange } }) => (
                <SmartInput
                  config={field}
                  value={value}
                  onChange={onChange}
                  error={errors[field.name] as FieldError}
                  disabled={isSubmitting}
                />
              )}
            />
          ))}

          <div className="flex items-center justify-between pt-4">
            <div className="flex items-center gap-2">
              {!isValid && Object.keys(errors).length > 0 && (
                <Badge variant="destructive" className="text-xs">
                  {Object.keys(errors).length} error(s)
                </Badge>
              )}
              {isValid && (
                <Badge variant="secondary" className="text-xs">
                  Form valid
                </Badge>
              )}
            </div>
            
            <div className="flex gap-2">
              <Button
                type="button"
                variant="outline"
                onClick={() => reset()}
                disabled={isSubmitting}
              >
                Reset
              </Button>
              <Button
                type="submit"
                disabled={isSubmitting || !isValid}
              >
                {isSubmitting ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    Submitting...
                  </>
                ) : (
                  config.submitText || 'Submit'
                )}
              </Button>
            </div>
          </div>
        </form>
      </CardContent>
    </Card>
  )
}