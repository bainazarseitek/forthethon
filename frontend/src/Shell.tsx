import { Link } from "react-router-dom";

export function caseNumber(id: string) {
  return id
    .slice(0, 8)
    .toUpperCase()
    .replace(/(.{4})(.{4})/, "$1·$2");
}

export function Shell({
  children,
  trail,
  actions,
}: {
  children: React.ReactNode;
  trail?: React.ReactNode;
  actions?: React.ReactNode;
}) {
  return (
    <>
      <a
        className="skip"
        href="#main"
        onClick={(e) => {
          e.preventDefault();
          const main = document.getElementById("main");
          main?.setAttribute("tabindex", "-1");
          main?.focus();
        }}
      >
        Перейти к содержимому
      </a>
      <header className="topbar">
        <Link to="/" className="brand" aria-label="Forte Review, на главную">
          <span className="brand-mark" aria-hidden="true" />
          <span className="brand-word">
            forte <span>review</span>
          </span>
        </Link>
        {trail && <div className="trail">{trail}</div>}
        <div className="topbar-actions">{actions}</div>
      </header>
      {children}
    </>
  );
}
