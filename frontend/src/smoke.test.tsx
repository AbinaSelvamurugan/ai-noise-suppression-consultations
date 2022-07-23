import { describe, expect, it, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import App from './App'

describe('ClearVoice navigation', () => {
  it('renders status and accessible navigation without assuming connectivity', () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('offline')))
    render(<App />)
    expect(screen.getByText('Welcome to ClearVoice')).toBeTruthy()
    expect(screen.getByRole('navigation', {name: 'Main navigation'})).toBeTruthy()
    expect(screen.getByText('No clinical certification or verbatim-transcription guarantee.', {exact: false})).toBeTruthy()
  })
})
