import React from "react";

export interface DownloadButtonsProps {
  onDownloadPDF?: () => void;
  onDownloadWord?: () => void;
  disabled?: boolean;
  className?: string;
  style?: React.CSSProperties;
}

export const DownloadButtons: React.FC<DownloadButtonsProps> = ({
  onDownloadPDF,
  onDownloadWord,
  disabled = false,
  className = "",
  style = {},
}) => {
  return (
    <div className={`download-buttons ${className}`} style={style}>
      <button
        type="button"
        className="download-buttons__btn download-buttons__btn--pdf"
        onClick={onDownloadPDF}
        disabled={disabled}
      >
        Download as PDF
      </button>
      <button
        type="button"
        className="download-buttons__btn download-buttons__btn--word"
        onClick={onDownloadWord}
        disabled={disabled}
      >
        Download as Word
      </button>
    </div>
  );
};

export default DownloadButtons;