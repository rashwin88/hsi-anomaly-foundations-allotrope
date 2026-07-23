// Placeholder for sidebar destinations not yet implemented.
// Each gets a brief, accurate description per storyboard-spec § 5.

interface PlaceholderPageProps {
  title: string;
  body: string;
  step?: string;
}

export function PlaceholderPage({ title, body, step }: PlaceholderPageProps) {
  return (
    <div className="placeholder">
      <h2 className="placeholder__title">{title}</h2>
      <p className="placeholder__body">{body}</p>
      {step && <p className="placeholder__step">Coming in {step}.</p>}
    </div>
  );
}
