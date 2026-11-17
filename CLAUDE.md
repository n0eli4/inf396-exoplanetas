# Reglas del proyecto

## Commits

- **Nunca** agregar a Claude como coautor en un commit: ni `Co-Authored-By: Claude ...`
  ni ninguna otra línea de atribución a Claude en mensajes de commit o descripciones de PR.
- Después de hacer cada commit, verificar con `git log -1 --format=%B` que el mensaje no
  contenga "Co-Authored-By" ni "Claude", y **avisar explícitamente al usuario** el resultado
  de esa verificación.
- Los commits van a nombre de Noelia (configuración de git de este repo).
