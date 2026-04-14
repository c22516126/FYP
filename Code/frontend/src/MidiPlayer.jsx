import { useEffect, useState } from "react"

function MidiPlayer({ midiUrl }) {

    const [activeNotes, setActiveNotes] = useState(new Set())
    useEffect(() => {
        const player = document.querySelector('midi-player')
        const visualizer = document.querySelector('midi-visualizer')

        if (player && visualizer) {
            player.addVisualizer(visualizer)
        }

        const handleNote = (e) => {
            const pitch = e.detail.note.pitch

            setActiveNotes(prev => {
                const updated = new Set(prev)
                updated.add(pitch)

                setTimeout(() => {
                    setActiveNotes(current => {
                        const next = new Set(current)
                        next.delete(pitch)
                        return next
                    })
                }, 200)

                return updated
            })
        }

        if (player) {
            player.addEventListener('note', handleNote)
        }

        return () => {
            if (player) {
                player.removeEventListener('note', handleNote)
            }
        }

    }, [midiUrl])


    const noteNames = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    const getNoteName = (midi) => {
        const octave = Math.floor(midi / 12) - 1
        return noteNames[midi % 12] + octave
    }

    const isBlackKey = (midi) => {
        return [1, 3, 6, 8, 10].includes(midi % 12)
    }

    const keys = Array.from({ length: 60 }, (_, i) => 84 - i)

    return (
        <div id="visContainer">
            <midi-player src={midiUrl} sound-font />
            <div id="midiVisualizer">

                {/* player + roll */}
                <div id="playerRoll">
                    <midi-visualizer
                        src={midiUrl}
                        type="waterfall"
                    />
                </div>

            </div>
        </div>
    )
}

export default MidiPlayer